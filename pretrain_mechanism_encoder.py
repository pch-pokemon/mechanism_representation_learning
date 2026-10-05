# -*- coding: utf-8 -*-

"""
Mechanism-supervised pretraining of a six-channel oxidation-state encoder.

Paper-release implementation
----------------------------
This script implements the pretraining procedure described in the manuscript:

    structured material variables
        -> shared MLP encoder
        -> six independent 64-D mechanism heads

Each mechanism head is supervised by the corresponding literature-derived
64-dimensional reference representation. Missing / not_reported mechanism
channels are masked from the loss. The six valid channel losses are averaged,
so channels with greater reporting coverage do not dominate the objective.

Important design choices fixed for the manuscript release
---------------------------------------------------------
- Six mechanism channels are always used.
- Reference representation dimension is fixed at 64.
- The train / validation / test membership is read from one fixed split manifest.
- Feature scaling is fitted on the training split only.
- Pretraining uses cosine-distance supervision only.
- No auxiliary classification loss is used in the manuscript pretraining model.
- The validation split is used for model selection.
- The test split is never evaluated by this script; final test evaluation is
  performed separately by evaluate_pretraining.py.

Expected reference-space layout
-------------------------------
<reference_root>/
    channel_1/reference_embeddings_64d.csv
    channel_2/reference_embeddings_64d.csv
    ...
    channel_6/reference_embeddings_64d.csv

Each reference CSV is produced by build_mechanism_embeddings.py and contains
four alignment keys, embedding_valid, and emb_0 ... emb_63.
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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Fixed manuscript definitions
# =============================================================================

FEATURE_COLS: Tuple[str, ...] = (
    "solu_temp", "solu_time",
    "middle_temp", "middle_time",
    "aging_temp", "aging_time",
    "Co", "Al", "W", "Ni", "Cr", "Mo", "Fe", "Nb", "C", "Hf",
    "Si", "Ta", "Ti", "Y", "V", "B", "Zr", "Mn", "Sc", "La", "Re",
    "Temperature",
    "E_Time",
)

ALIGN_KEYS: Tuple[str, ...] = (
    "Ref_source_num",
    "sample_id",
    "alloy_id",
    "condition_id",
)

CHANNEL_NAMES: Dict[int, str] = {
    1: "protective_scale_formation",
    2: "internal_oxidation_and_selective_oxidation",
    3: "nonprotective_or_transient_oxide_formation",
    4: "transport_and_defect_kinetics",
    5: "scale_failure_and_spallation",
    6: "alloying_element_interaction",
}

SELECTED_CHANNELS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
EMBED_DIM = 64


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class TrainConfig:
    hidden_dims: Tuple[int, ...]
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    random_seed: int = 42
    num_workers: int = 0
    log_interval: int = 20


def load_training_config(path: Path) -> TrainConfig:
    """Load the fixed hyperparameters used for the final manuscript model."""
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Training config must be a JSON object.")

    required = {
        "hidden_dims",
        "dropout",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "patience",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Training config is missing fields: {missing}")

    hidden_dims = tuple(int(x) for x in payload["hidden_dims"])
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain positive integers.")

    cfg = TrainConfig(
        hidden_dims=hidden_dims,
        dropout=float(payload["dropout"]),
        batch_size=int(payload["batch_size"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        patience=int(payload["patience"]),
        random_seed=int(payload.get("random_seed", 42)),
        num_workers=int(payload.get("num_workers", 0)),
        log_interval=int(payload.get("log_interval", 20)),
    )

    if not 0.0 <= cfg.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if cfg.batch_size <= 1:
        raise ValueError("batch_size must be greater than 1 because BatchNorm is used.")
    if cfg.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if cfg.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if cfg.patience <= 0:
        raise ValueError("patience must be positive.")

    return cfg


# =============================================================================
# Reproducibility helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested)


def normalize_ids(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Missing required identifier column: {column}")
        result[column] = result[column].fillna("").astype(str).str.strip()
        if result[column].eq("").any():
            examples = result.loc[result[column].eq(""), list(columns)].head(10)
            raise ValueError(
                f"Empty values found in identifier column '{column}'.\n{examples}"
            )
    return result


# =============================================================================
# Data loading and fixed split attachment
# =============================================================================

def load_structured_data(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Structured dataset not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df = normalize_ids(df, ALIGN_KEYS)

    if df.duplicated(list(ALIGN_KEYS)).any():
        examples = df.loc[
            df.duplicated(list(ALIGN_KEYS), keep=False), list(ALIGN_KEYS)
        ].head(20)
        raise ValueError(f"Duplicate alignment keys in structured dataset:\n{examples}")

    for column in FEATURE_COLS:
        if column not in df.columns:
            raise ValueError(f"Missing structured input feature: {column}")
        # In the curated structured dataset, absence of an element or optional
        # heat-treatment step is represented by zero.
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df


def load_split_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    split_df = pd.read_csv(path, dtype=str)
    split_df = normalize_ids(split_df, ALIGN_KEYS)

    if "split" not in split_df.columns:
        raise ValueError("Split manifest must contain a 'split' column.")

    split_df["split"] = split_df["split"].fillna("").str.strip().str.lower()
    allowed = {"train", "val", "test"}
    invalid = ~split_df["split"].isin(allowed)
    if invalid.any():
        examples = split_df.loc[invalid, list(ALIGN_KEYS) + ["split"]].head(20)
        raise ValueError(f"Split values must be train/val/test.\n{examples}")

    if split_df.duplicated(list(ALIGN_KEYS)).any():
        raise ValueError("Split manifest contains duplicate sample-condition identities.")

    return split_df[list(ALIGN_KEYS) + ["split"]].copy()


def reference_csv_path(reference_root: Path, channel: int) -> Path:
    return reference_root / f"channel_{channel}" / "reference_embeddings_64d.csv"


def attach_reference_spaces(
    structured_df: pd.DataFrame,
    reference_root: Path,
) -> pd.DataFrame:
    """Attach all six reference spaces through strict four-key alignment."""
    merged = structured_df.copy()
    emb_cols = [f"emb_{i}" for i in range(EMBED_DIM)]

    for channel in SELECTED_CHANNELS:
        path = reference_csv_path(reference_root, channel)
        if not path.exists():
            raise FileNotFoundError(f"Channel-{channel} reference CSV not found: {path}")

        ref = pd.read_csv(path)
        ref = normalize_ids(ref, ALIGN_KEYS)

        if ref.duplicated(list(ALIGN_KEYS)).any():
            raise ValueError(f"Duplicate keys in channel-{channel} reference CSV.")

        required = list(ALIGN_KEYS) + ["embedding_valid"] + emb_cols
        missing = [column for column in required if column not in ref.columns]
        if missing:
            raise ValueError(
                f"Channel-{channel} reference CSV is missing columns: {missing[:10]}"
            )

        keep = ref[required].copy()
        rename = {
            column: f"ch{channel}__{column}"
            for column in keep.columns
            if column not in ALIGN_KEYS
        }
        keep = keep.rename(columns=rename)

        before = len(merged)
        merged = merged.merge(
            keep,
            on=list(ALIGN_KEYS),
            how="left",
            validate="one_to_one",
        )
        if len(merged) != before:
            raise RuntimeError("Reference merge unexpectedly changed row count.")

        check_cols = [f"ch{channel}__emb_{i}" for i in range(EMBED_DIM)]
        missing_rows = merged[check_cols].isna().any(axis=1)
        if missing_rows.any():
            examples = merged.loc[missing_rows, list(ALIGN_KEYS)].head(20)
            raise ValueError(
                f"Channel-{channel} failed strict alignment for "
                f"{int(missing_rows.sum())} rows.\n{examples}"
            )

    return merged


def attach_fixed_split(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    result = df.merge(
        split_df,
        on=list(ALIGN_KEYS),
        how="left",
        validate="one_to_one",
    )

    if result["split"].isna().any():
        examples = result.loc[result["split"].isna(), list(ALIGN_KEYS)].head(20)
        raise ValueError(f"Rows missing from fixed split manifest:\n{examples}")

    return result


# =============================================================================
# Prepared arrays
# =============================================================================

@dataclass
class PreparedData:
    df: pd.DataFrame
    scaler: StandardScaler
    x_scaled: np.ndarray
    channel_targets: Dict[int, np.ndarray]
    channel_masks: Dict[int, np.ndarray]


def prepare_data(
    structured_data: Path,
    sheet_name: str,
    reference_root: Path,
    split_csv: Path,
) -> PreparedData:
    df = load_structured_data(structured_data, sheet_name)
    df = attach_reference_spaces(df, reference_root)
    df = attach_fixed_split(df, load_split_manifest(split_csv))

    x = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    train_mask = df["split"].eq("train").to_numpy()

    if not train_mask.any():
        raise ValueError("Fixed split contains no training rows.")
    if not df["split"].eq("val").any():
        raise ValueError("Fixed split contains no validation rows.")
    if not df["split"].eq("test").any():
        raise ValueError("Fixed split contains no test rows.")

    scaler = StandardScaler()
    scaler.fit(x[train_mask])
    x_scaled = scaler.transform(x).astype(np.float32)

    channel_targets: Dict[int, np.ndarray] = {}
    channel_masks: Dict[int, np.ndarray] = {}

    for channel in SELECTED_CHANNELS:
        cols = [f"ch{channel}__emb_{i}" for i in range(EMBED_DIM)]
        channel_targets[channel] = df[cols].to_numpy(dtype=np.float32)

        valid = pd.to_numeric(
            df[f"ch{channel}__embedding_valid"], errors="raise"
        ).astype(int)
        if not valid.isin([0, 1]).all():
            raise ValueError(f"Invalid embedding_valid values in channel {channel}.")
        channel_masks[channel] = valid.eq(1).to_numpy(dtype=bool)

        print(
            f"Channel {channel}: reported mechanism rows = "
            f"{int(channel_masks[channel].sum())}/{len(df)}"
        )

    return PreparedData(
        df=df,
        scaler=scaler,
        x_scaled=x_scaled,
        channel_targets=channel_targets,
        channel_masks=channel_masks,
    )


# =============================================================================
# Dataset
# =============================================================================

class MechanismDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        row_indices: np.ndarray,
        channel_targets: Mapping[int, np.ndarray],
        channel_masks: Mapping[int, np.ndarray],
    ) -> None:
        self.x = torch.as_tensor(x[row_indices], dtype=torch.float32)
        self.targets = {
            str(channel): torch.as_tensor(values[row_indices], dtype=torch.float32)
            for channel, values in channel_targets.items()
        }
        self.masks = {
            str(channel): torch.as_tensor(mask[row_indices], dtype=torch.bool)
            for channel, mask in channel_masks.items()
        }

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "x": self.x[index],
            "targets": {key: value[index] for key, value in self.targets.items()},
            "masks": {key: value[index] for key, value in self.masks.items()},
        }


# =============================================================================
# Model
# =============================================================================

class SharedMLPEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty.")

        layers: List[nn.Module] = []
        previous = in_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, int(width)),
                    nn.BatchNorm1d(int(width)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(width)

        self.net = nn.Sequential(*layers)
        self.out_dim = previous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiChannelMechanismEncoder(nn.Module):
    """Shared material encoder with six independent 64-D mechanism heads."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = SharedMLPEncoder(in_dim, hidden_dims, dropout)
        self.embedding_heads = nn.ModuleDict(
            {
                str(channel): nn.Linear(self.encoder.out_dim, EMBED_DIM)
                for channel in SELECTED_CHANNELS
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encoder(x)
        return {
            key: head(latent)
            for key, head in self.embedding_heads.items()
        }


# =============================================================================
# Cosine-distance objective
# =============================================================================

def masked_channel_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return None

    cosine = F.cosine_similarity(
        prediction[mask],
        target[mask],
        dim=1,
        eps=1e-12,
    )
    return 1.0 - cosine.mean()


def compute_multichannel_loss(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor | None, Dict[str, float]]:
    losses: List[torch.Tensor] = []
    components: Dict[str, float] = {}

    for channel in SELECTED_CHANNELS:
        key = str(channel)
        loss = masked_channel_cosine_loss(
            prediction=predictions[key],
            target=batch["targets"][key].to(device),
            mask=batch["masks"][key].to(device),
        )
        if loss is None:
            continue

        losses.append(loss)
        components[f"loss_ch{channel}"] = float(loss.detach().cpu())

    if not losses:
        return None, components

    total = torch.stack(losses).mean()
    components["loss_total"] = float(total.detach().cpu())
    return total, components


# =============================================================================
# Validation
# =============================================================================

def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    losses: List[float] = []
    true_vectors: Dict[int, List[np.ndarray]] = {
        channel: [] for channel in SELECTED_CHANNELS
    }
    pred_vectors: Dict[int, List[np.ndarray]] = {
        channel: [] for channel in SELECTED_CHANNELS
    }

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            prediction = model(x)
            loss, _ = compute_multichannel_loss(prediction, batch, device)
            if loss is not None:
                losses.append(float(loss.cpu()))

            for channel in SELECTED_CHANNELS:
                key = str(channel)
                mask = batch["masks"][key].bool()
                if int(mask.sum().item()) == 0:
                    continue

                true_vectors[channel].append(
                    batch["targets"][key][mask].numpy()
                )
                pred_vectors[channel].append(
                    prediction[key][mask.to(device)].cpu().numpy()
                )

    metrics: Dict[str, float] = {
        "val_loss": float(np.mean(losses)) if losses else float("nan")
    }

    channel_means: List[float] = []

    for channel in SELECTED_CHANNELS:
        if not true_vectors[channel]:
            metrics[f"ch{channel}_n_valid"] = 0
            metrics[f"ch{channel}_cosine_mean"] = float("nan")
            continue

        target = np.vstack(true_vectors[channel])
        prediction = np.vstack(pred_vectors[channel])

        target_norm = target / np.clip(
            np.linalg.norm(target, axis=1, keepdims=True), 1e-12, None
        )
        pred_norm = prediction / np.clip(
            np.linalg.norm(prediction, axis=1, keepdims=True), 1e-12, None
        )
        cosine = np.sum(target_norm * pred_norm, axis=1)

        mean_cosine = float(np.mean(cosine))
        channel_means.append(mean_cosine)

        metrics.update(
            {
                f"ch{channel}_n_valid": int(len(cosine)),
                f"ch{channel}_cosine_mean": mean_cosine,
                f"ch{channel}_cosine_median": float(np.median(cosine)),
                f"ch{channel}_cosine_p25": float(np.percentile(cosine, 25)),
            }
        )

    if len(channel_means) != len(SELECTED_CHANNELS):
        raise RuntimeError(
            "At least one mechanism channel has no valid validation samples; "
            "the channel-averaged validation CS cannot be computed."
        )

    metrics["mean_channel_cosine"] = float(np.mean(channel_means))
    return metrics


# =============================================================================
# Training
# =============================================================================

def mean_component_dict(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


def train_model(
    prepared: PreparedData,
    cfg: TrainConfig,
    output_dir: Path,
    device: torch.device,
) -> None:
    train_idx = np.flatnonzero(prepared.df["split"].eq("train").to_numpy())
    val_idx = np.flatnonzero(prepared.df["split"].eq("val").to_numpy())

    train_ds = MechanismDataset(
        prepared.x_scaled,
        train_idx,
        prepared.channel_targets,
        prepared.channel_masks,
    )
    val_ds = MechanismDataset(
        prepared.x_scaled,
        val_idx,
        prepared.channel_targets,
        prepared.channel_masks,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MultiChannelMechanismEncoder(
        in_dim=len(FEATURE_COLS),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_score = -float("inf")
    best_epoch = -1
    wait = 0
    history: List[Dict[str, Any]] = []

    print("=" * 88)
    print("Mechanism-supervised pretraining")
    print("=" * 88)
    print(f"Train / validation / test rows: "
          f"{prepared.df['split'].eq('train').sum()} / "
          f"{prepared.df['split'].eq('val').sum()} / "
          f"{prepared.df['split'].eq('test').sum()}")
    print(f"Hidden dimensions:             {cfg.hidden_dims}")
    print(f"Mechanism embedding dimension: {EMBED_DIM}")
    print(f"Training objective:            cosine distance only")
    print(f"Model-selection metric:        mean validation CS across six channels")
    print(f"Device:                        {device}")
    print("=" * 88)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_components: List[Dict[str, float]] = []
        skipped_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)

            # BatchNorm requires more than one training sample.
            if x.shape[0] < 2:
                skipped_batches += 1
                continue

            predictions = model(x)
            loss, components = compute_multichannel_loss(
                predictions, batch, device
            )

            if loss is None:
                skipped_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_components.append(components)

        train_metrics = mean_component_dict(train_components)
        val_metrics = evaluate_validation(model, val_loader, device)
        score = float(val_metrics["mean_channel_cosine"])

        row: Dict[str, Any] = {
            "epoch": epoch,
            "skipped_train_batches": skipped_batches,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        if epoch == 1 or epoch % cfg.log_interval == 0:
            channel_text = " | ".join(
                f"ch{channel}={val_metrics[f'ch{channel}_cosine_mean']:.4f}"
                for channel in SELECTED_CHANNELS
            )
            print(
                f"Epoch {epoch:04d} | "
                f"mean val CS={score:.4f} | "
                f"val loss={val_metrics['val_loss']:.4f} | "
                f"{channel_text}"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            wait = 0
            state_cpu = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            torch.save(state_cpu, output_dir / "best_model_state.pt")
        else:
            wait += 1

        if wait >= cfg.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, mean val CS={best_score:.6f}."
            )
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        output_dir / "pretrain_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if best_epoch < 0:
        raise RuntimeError("No valid checkpoint was saved.")

    # Reload best checkpoint and record its validation metrics.
    best_state = torch.load(
        output_dir / "best_model_state.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)
    best_metrics = evaluate_validation(model, val_loader, device)

    with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_mean_validation_cosine": best_score,
                **best_metrics,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    joblib.dump(prepared.scaler, output_dir / "feature_scaler.joblib")

    resolved = {
        **asdict(cfg),
        "hidden_dims": list(cfg.hidden_dims),
        "feature_cols": list(FEATURE_COLS),
        "alignment_keys": list(ALIGN_KEYS),
        "selected_channels": list(SELECTED_CHANNELS),
        "channel_names": {str(k): v for k, v in CHANNEL_NAMES.items()},
        "embedding_dimension": EMBED_DIM,
        "objective": "mean masked channel-wise cosine distance",
        "model_selection_metric": "mean validation cosine similarity across six channels",
        "test_used_during_training": False,
        "best_epoch": best_epoch,
        "best_mean_validation_cosine": best_score,
    }

    with (output_dir / "resolved_pretraining_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(resolved, file, indent=2, ensure_ascii=False)

    # Save the fixed split identity used by this run without any model outputs.
    prepared.df[list(ALIGN_KEYS) + ["split"]].to_csv(
        output_dir / "data_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 88)
    print(f"Best epoch:                   {best_epoch}")
    print(f"Best mean validation CS:      {best_score:.6f}")
    print(f"Saved outputs to:             {output_dir.resolve()}")
    print("=" * 88)


# =============================================================================
# Command line
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the final six-channel mechanism-supervised encoder."
    )
    parser.add_argument(
        "--structured-data",
        required=True,
        type=Path,
        help="Curated structured sample-condition dataset (Excel).",
    )
    parser.add_argument(
        "--sheet-name",
        default="Sheet3",
        help="Structured-data worksheet used for mechanism pretraining.",
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        type=Path,
        help="Root directory produced by build_mechanism_embeddings.py.",
    )
    parser.add_argument(
        "--split-csv",
        required=True,
        type=Path,
        help="Fixed 60:20:20 train/validation/test split manifest.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="JSON file containing the final pretraining hyperparameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pretraining"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_training_config(args.config)

    set_seed(cfg.random_seed)
    device = resolve_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(
        structured_data=args.structured_data,
        sheet_name=args.sheet_name,
        reference_root=args.reference_root,
        split_csv=args.split_csv,
    )

    train_model(
        prepared=prepared,
        cfg=cfg,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
