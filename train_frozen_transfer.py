# -*- coding: utf-8 -*-

"""
Frozen transfer from the mechanism-supervised encoder to oxidation-kinetics prediction.

Paper-release implementation
----------------------------
The pretrained shared encoder and all six mechanism heads are kept fixed.
Only a newly introduced kinetics-prediction MLP is optimized.

For each structured sample-condition input x:

    x
      -> frozen pretrained shared encoder -> latent material representation z
           |-> frozen mechanism head 1 -> 64-D mechanism representation
           |-> frozen mechanism head 2 -> 64-D mechanism representation
           |-> ...
           |-> frozen mechanism head 6 -> 64-D mechanism representation
           |
           -> trainable kinetics head -> predicted lg(kp)

The six mechanism vectors remain parallel outputs and are NOT concatenated
into the kinetics head.

Important manuscript-release choices
------------------------------------
- The pretrained encoder and six mechanism heads are frozen throughout transfer.
- The exact feature scaler saved during mechanism pretraining is reused.
- Kinetics training uses unweighted mean squared error only.
- No mechanism-alignment loss is added during transfer.
- No auxiliary classification loss is used in the manuscript transfer model.
- The fixed 60:20:20 split identity from mechanism pretraining is reused.
- The validation subset is used for checkpoint selection.
- The test subset is never evaluated by this script; final test evaluation is
  performed separately by evaluate_kinetics.py.
- Best-epoch selection is fixed to validation RMSE.

Expected pretraining run directory
----------------------------------
<pretrain_run>/
    best_model_state.pt
    feature_scaler.joblib
    resolved_pretraining_config.json
    data_split.csv

Expected transfer configuration
-------------------------------
A JSON file containing the final kinetics-head hyperparameters, e.g.

{
  "hidden_dims": [128, 64, 32],
  "dropout": 0.10,
  "batch_size": 32,
  "epochs": 3000,
  "learning_rate": 0.001,
  "weight_decay": 1e-5,
  "patience": 200,
  "scheduler_factor": 0.5,
  "scheduler_patience": 60,
  "scheduler_min_lr": 1e-7,
  "gradient_clip_norm": 5.0,
  "random_seed": 42,
  "num_workers": 0,
  "log_interval": 20
}

The numerical values above are examples only. The repository config should contain
the exact final hyperparameters used for the manuscript results.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pretrain_mechanism_encoder import (
    ALIGN_KEYS,
    CHANNEL_NAMES,
    FEATURE_COLS,
    MultiChannelMechanismEncoder,
    SELECTED_CHANNELS,
    load_split_manifest,
    normalize_ids,
    resolve_device,
)


TARGET_COL = "lgkp"


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class TransferConfig:
    hidden_dims: Tuple[int, ...]
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_lr: float
    gradient_clip_norm: float
    random_seed: int = 42
    num_workers: int = 0
    log_interval: int = 20


def load_transfer_config(path: Path) -> TransferConfig:
    if not path.exists():
        raise FileNotFoundError(f"Transfer config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Transfer config must be a JSON object.")

    required = {
        "hidden_dims",
        "dropout",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "patience",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_min_lr",
        "gradient_clip_norm",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Transfer config is missing fields: {missing}")

    hidden_dims = tuple(int(x) for x in payload["hidden_dims"])
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain positive integers.")

    cfg = TransferConfig(
        hidden_dims=hidden_dims,
        dropout=float(payload["dropout"]),
        batch_size=int(payload["batch_size"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        patience=int(payload["patience"]),
        scheduler_factor=float(payload["scheduler_factor"]),
        scheduler_patience=int(payload["scheduler_patience"]),
        scheduler_min_lr=float(payload["scheduler_min_lr"]),
        gradient_clip_norm=float(payload["gradient_clip_norm"]),
        random_seed=int(payload.get("random_seed", 42)),
        num_workers=int(payload.get("num_workers", 0)),
        log_interval=int(payload.get("log_interval", 20)),
    )

    if not 0.0 <= cfg.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if cfg.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if cfg.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if cfg.patience <= 0:
        raise ValueError("patience must be positive.")
    if not 0.0 < cfg.scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must be in (0, 1).")
    if cfg.scheduler_patience <= 0:
        raise ValueError("scheduler_patience must be positive.")
    if cfg.scheduler_min_lr <= 0:
        raise ValueError("scheduler_min_lr must be positive.")
    if cfg.gradient_clip_norm < 0:
        raise ValueError("gradient_clip_norm must be non-negative.")

    return cfg


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data
# =============================================================================

def load_kinetics_data(
    path: Path,
    sheet_name: str,
    split_csv: Path,
) -> pd.DataFrame:
    """
    Load all kinetics-labelled rows and attach the fixed manuscript split identity.

    Split membership is never regenerated here.
    """
    if not path.exists():
        raise FileNotFoundError(f"Kinetics dataset not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df = normalize_ids(df, ALIGN_KEYS)

    if df.duplicated(list(ALIGN_KEYS)).any():
        examples = df.loc[
            df.duplicated(list(ALIGN_KEYS), keep=False),
            list(ALIGN_KEYS),
        ].head(20)
        raise ValueError(f"Duplicate alignment keys in kinetics dataset:\n{examples}")

    required = [*FEATURE_COLS, TARGET_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Kinetics dataset is missing columns: {missing}")

    # Numeric structured inputs use the same convention as pretraining.
    for column in FEATURE_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    # Keep only usable kinetic labels.
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid_target = np.isfinite(df[TARGET_COL].to_numpy(dtype=float))

    if not valid_target.all():
        print(
            f"Dropping {int((~valid_target).sum())}/{len(df)} rows "
            f"without a usable {TARGET_COL} label."
        )
        df = df.loc[valid_target].copy()

    split_df = load_split_manifest(split_csv)
    df = df.merge(
        split_df,
        on=list(ALIGN_KEYS),
        how="left",
        validate="one_to_one",
    )

    if df["split"].isna().any():
        examples = df.loc[
            df["split"].isna(),
            list(ALIGN_KEYS),
        ].head(20)
        raise ValueError(
            "Some kinetics-labelled records do not have a fixed split assignment.\n"
            f"{examples}"
        )

    for split in ("train", "val", "test"):
        if not df["split"].eq(split).any():
            raise ValueError(f"No kinetics-labelled rows are assigned to split={split!r}.")

    return df.reset_index(drop=True)


class KineticsDataset(Dataset):
    def __init__(
        self,
        x_scaled: np.ndarray,
        y: np.ndarray,
        row_indices: np.ndarray,
    ) -> None:
        self.x = torch.as_tensor(
            x_scaled[row_indices],
            dtype=torch.float32,
        )
        self.y = torch.as_tensor(
            y[row_indices],
            dtype=torch.float32,
        )
        self.global_index = torch.as_tensor(
            row_indices,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "global_index": self.global_index[index],
        }


# =============================================================================
# Pretrained mechanism model
# =============================================================================

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return payload


def load_pretrained_backbone(
    pretrain_run: Path,
    device: torch.device,
) -> MultiChannelMechanismEncoder:
    """
    Reconstruct and load the exact mechanism-pretrained encoder.
    """
    config = read_json(
        pretrain_run / "resolved_pretraining_config.json"
    )

    hidden_dims = tuple(int(x) for x in config["hidden_dims"])
    dropout = float(config["dropout"])

    backbone = MultiChannelMechanismEncoder(
        in_dim=len(FEATURE_COLS),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    checkpoint = pretrain_run / "best_model_state.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint}")

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    backbone.load_state_dict(state_dict, strict=True)

    # Freeze every pretrained parameter.
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    backbone.to(device)
    backbone.eval()

    return backbone


# =============================================================================
# Frozen-transfer model
# =============================================================================

def build_mlp_head(
    in_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = in_dim

    for width in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous, int(width)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            ]
        )
        previous = int(width)

    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


class FrozenTransferKineticsModel(nn.Module):
    """
    Frozen mechanism-supervised backbone plus one trainable kinetics head.
    """

    def __init__(
        self,
        backbone: MultiChannelMechanismEncoder,
        kinetics_hidden_dims: Sequence[int],
        kinetics_dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        latent_dim = int(backbone.encoder.out_dim)

        self.kinetics_head = build_mlp_head(
            in_dim=latent_dim,
            hidden_dims=kinetics_hidden_dims,
            dropout=kinetics_dropout,
        )

    def train(self, mode: bool = True):
        """
        Keep the pretrained backbone in evaluation mode even while the new
        kinetics head is trained. This freezes BatchNorm running statistics and
        disables pretrained-backbone dropout.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        # No gradient graph is needed through the frozen representation.
        with torch.no_grad():
            latent = self.backbone.encoder(x)
            mechanism_vectors = {
                key: head(latent)
                for key, head in self.backbone.embedding_heads.items()
            }

        lgkp = self.kinetics_head(latent).squeeze(-1)

        return {
            "latent": latent,
            "mechanism_vectors": mechanism_vectors,
            "lgkp": lgkp,
        }


# =============================================================================
# Metrics
# =============================================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


@torch.no_grad()
def evaluate_loader(
    model: FrozenTransferKineticsModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()

    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        outputs = model(x)

        y_true_parts.append(
            batch["y"].cpu().numpy()
        )
        y_pred_parts.append(
            outputs["lgkp"].cpu().numpy()
        )
        index_parts.append(
            batch["global_index"].cpu().numpy()
        )

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    indices = np.concatenate(index_parts).astype(np.int64)

    metrics = regression_metrics(y_true, y_pred)
    metrics["MSE"] = float(mean_squared_error(y_true, y_pred))

    arrays = {
        "y_true": y_true,
        "y_pred": y_pred,
        "global_index": indices,
    }

    return metrics, arrays


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model: FrozenTransferKineticsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()

    losses: List[float] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        outputs = model(x)

        # Manuscript Eq. (3): unweighted mean squared error.
        loss = F.mse_loss(outputs["lgkp"], y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.kinetics_head.parameters(),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    if not losses:
        raise RuntimeError("Training loader produced no batches.")

    return float(np.mean(losses))


def build_prediction_frame(
    df: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    split_name: str,
) -> pd.DataFrame:
    indices = np.asarray(arrays["global_index"], dtype=np.int64)

    keep_columns = [
        *ALIGN_KEYS,
        "split",
        TARGET_COL,
    ]

    out = df.iloc[indices][keep_columns].copy().reset_index(drop=True)
    out["evaluated_split"] = split_name
    out["lgkp_true"] = np.asarray(arrays["y_true"], dtype=float)
    out["lgkp_pred"] = np.asarray(arrays["y_pred"], dtype=float)
    out["residual"] = out["lgkp_pred"] - out["lgkp_true"]
    out["abs_error"] = np.abs(out["residual"])
    return out


def train_frozen_transfer(
    df: pd.DataFrame,
    pretrain_run: Path,
    cfg: TransferConfig,
    output_dir: Path,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reuse the exact pretraining feature scaler.
    # ------------------------------------------------------------------
    scaler_path = pretrain_run / "feature_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Pretraining feature scaler not found: {scaler_path}"
        )

    scaler = joblib.load(scaler_path)

    x_raw = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x_raw).astype(np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    train_idx = np.flatnonzero(
        df["split"].eq("train").to_numpy()
    )
    val_idx = np.flatnonzero(
        df["split"].eq("val").to_numpy()
    )
    test_idx = np.flatnonzero(
        df["split"].eq("test").to_numpy()
    )

    train_ds = KineticsDataset(
        x_scaled=x_scaled,
        y=y,
        row_indices=train_idx,
    )
    val_ds = KineticsDataset(
        x_scaled=x_scaled,
        y=y,
        row_indices=val_idx,
    )

    common_loader = {
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        **common_loader,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader,
    )

    backbone = load_pretrained_backbone(
        pretrain_run=pretrain_run,
        device=device,
    )

    model = FrozenTransferKineticsModel(
        backbone=backbone,
        kinetics_hidden_dims=cfg.hidden_dims,
        kinetics_dropout=cfg.dropout,
    ).to(device)

    # Safety check: only the new kinetics head may be trainable.
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_names:
        raise RuntimeError("No trainable kinetics-head parameters were found.")

    if any(not name.startswith("kinetics_head.") for name in trainable_names):
        raise RuntimeError(
            "Frozen-transfer safety check failed: a pretrained parameter is trainable."
        )

    optimizer = torch.optim.AdamW(
        model.kinetics_head.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.scheduler_min_lr,
    )

    best_rmse = float("inf")
    best_epoch = -1
    wait = 0
    history: List[Dict[str, Any]] = []

    print("=" * 92)
    print("Frozen transfer for oxidation-kinetics prediction")
    print("=" * 92)
    print(
        f"Kinetics-labelled train / val / test rows: "
        f"{len(train_idx)} / {len(val_idx)} / {len(test_idx)}"
    )
    print(f"Kinetics-head hidden dims:       {cfg.hidden_dims}")
    print(f"Kinetics-head dropout:           {cfg.dropout}")
    print(f"Learning rate:                   {cfg.learning_rate}")
    print(f"Weight decay:                    {cfg.weight_decay}")
    print("Training objective:              unweighted MSE")
    print("Checkpoint selection:            validation RMSE")
    print("Pretrained backbone:             frozen")
    print("Test-set evaluation:             deferred to evaluate_kinetics.py")
    print(f"Device:                          {device}")
    print("=" * 92)

    for epoch in range(1, cfg.epochs + 1):
        train_mse = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )

        val_metrics, _ = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
        )

        val_rmse = float(val_metrics["RMSE"])
        if not np.isfinite(val_rmse):
            raise RuntimeError("Validation RMSE became non-finite.")

        scheduler.step(val_rmse)

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        history_row = {
            "epoch": epoch,
            "train_MSE": train_mse,
            "val_MSE": float(val_metrics["MSE"]),
            "val_R2": float(val_metrics["R2"]),
            "val_MAE": float(val_metrics["MAE"]),
            "val_RMSE": val_rmse,
            "learning_rate": current_lr,
        }
        history.append(history_row)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            wait = 0

            state_cpu = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            torch.save(
                state_cpu,
                output_dir / "best_transfer_model_state.pt",
            )

            torch.save(
                {
                    "model_state_dict": state_cpu,
                    "best_epoch": best_epoch,
                    "best_validation_RMSE": best_rmse,
                    "selection_metric": "validation_RMSE",
                    "kinetics_hidden_dims": list(cfg.hidden_dims),
                    "kinetics_dropout": cfg.dropout,
                    "pretrained_backbone_frozen": True,
                },
                output_dir / "best_transfer_checkpoint.pt",
            )
        else:
            wait += 1

        if epoch == 1 or epoch % cfg.log_interval == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train MSE={train_mse:.6f} | "
                f"val R2={val_metrics['R2']:.4f} | "
                f"MAE={val_metrics['MAE']:.4f} | "
                f"RMSE={val_rmse:.4f} | "
                f"lr={current_lr:.3g} | "
                f"wait={wait}/{cfg.patience}"
            )

        if wait >= cfg.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, "
                f"best validation RMSE={best_rmse:.6f}."
            )
            break

    if best_epoch < 0:
        raise RuntimeError("No transfer checkpoint was saved.")

    # ------------------------------------------------------------------
    # Reload best checkpoint and export only TRAIN + VALIDATION outputs.
    # The test split remains untouched here.
    # ------------------------------------------------------------------
    best_state = torch.load(
        output_dir / "best_transfer_model_state.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    train_metrics, train_arrays = evaluate_loader(
        model=model,
        loader=DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            **common_loader,
        ),
        device=device,
    )

    val_metrics, val_arrays = evaluate_loader(
        model=model,
        loader=val_loader,
        device=device,
    )

    pd.DataFrame(history).to_csv(
        output_dir / "transfer_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_df = pd.concat(
        [
            build_prediction_frame(
                df=df,
                arrays=train_arrays,
                split_name="train",
            ),
            build_prediction_frame(
                df=df,
                arrays=val_arrays,
                split_name="val",
            ),
        ],
        ignore_index=True,
    )

    prediction_df.to_csv(
        output_dir / "train_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Preserve only the split identity; no test predictions are exported here.
    df[list(ALIGN_KEYS) + ["split"]].to_csv(
        output_dir / "data_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_payload = {
        "best_epoch": best_epoch,
        "selection_metric": "validation_RMSE",
        "best_validation_RMSE": best_rmse,
        "pretrained_backbone_frozen": True,
        "training_objective": "unweighted_MSE",
        "mechanism_alignment_loss": False,
        "auxiliary_classification_loss": False,
        "quality_weighting": False,
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "train": train_metrics,
        "validation": val_metrics,
        "test": "reserved_for_evaluate_kinetics.py",
    }

    with (
        output_dir / "transfer_training_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    config_payload = asdict(cfg)
    config_payload.update(
        {
            "target_col": TARGET_COL,
            "feature_cols": list(FEATURE_COLS),
            "alignment_keys": list(ALIGN_KEYS),
            "mechanism_channels": {
                str(channel): CHANNEL_NAMES[channel]
                for channel in SELECTED_CHANNELS
            },
            "pretrained_backbone_frozen": True,
            "training_objective": "unweighted_MSE",
            "checkpoint_selection": "validation_RMSE",
            "test_evaluated_during_training": False,
        }
    )

    with (
        output_dir / "resolved_transfer_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 92)
    print(f"Best epoch:                {best_epoch}")
    print(f"Best validation RMSE:      {best_rmse:.6f}")
    print(
        f"Validation: R2={val_metrics['R2']:.4f}, "
        f"MAE={val_metrics['MAE']:.4f}, "
        f"RMSE={val_metrics['RMSE']:.4f}"
    )
    print("Held-out test set remains unevaluated.")
    print(f"Outputs:                   {output_dir.resolve()}")
    print("=" * 92)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen-transfer oxidation-kinetics head "
            "using the fixed manuscript split."
        )
    )

    parser.add_argument(
        "--kinetics-data",
        required=True,
        type=Path,
        help="Curated kinetics dataset containing structured inputs and lgkp.",
    )
    parser.add_argument(
        "--sheet-name",
        default="Sheet1",
    )
    parser.add_argument(
        "--pretrain-run",
        required=True,
        type=Path,
        help="Final mechanism-pretraining run directory.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Final frozen-transfer hyperparameter JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/frozen_transfer"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(
        load_transfer_config(args.config).random_seed
    )
    device = resolve_device(args.device)

    cfg = load_transfer_config(args.config)

    pretrain_run = args.pretrain_run.resolve()
    if not pretrain_run.exists():
        raise FileNotFoundError(
            f"Pretraining run directory not found: {pretrain_run}"
        )

    # Reuse the exact fixed split saved by mechanism pretraining.
    split_csv = pretrain_run / "data_split.csv"

    df = load_kinetics_data(
        path=args.kinetics_data,
        sheet_name=args.sheet_name,
        split_csv=split_csv,
    )

    train_frozen_transfer(
        df=df,
        pretrain_run=pretrain_run,
        cfg=cfg,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()