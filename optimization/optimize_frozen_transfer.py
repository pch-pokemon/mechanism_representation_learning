#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optuna hyperparameter optimization for frozen oxidation-kinetics transfer.

Paper-release implementation
----------------------------
Only the newly introduced kinetics-prediction head is optimized. The pretrained
shared encoder and all six mechanism heads remain frozen for every trial.

Search parameters
-----------------
- kinetics-head depth / hidden-layer widths
- kinetics-head dropout
- learning rate
- weight decay

Fixed choices
-------------
- frozen pretrained backbone
- batch size = 32
- unweighted MSE training objective
- validation RMSE for early stopping, pruning, and Optuna trial ranking
- no mechanism-alignment loss
- no auxiliary classification loss
- no kinetic-label quality weighting
- TEST data are never used for trial ranking, pruning, early stopping, or
  hyperparameter selection

The best hyperparameter set is exported as a JSON file directly compatible with
src/train_frozen_transfer.py.

Typical use
-----------
python optimization/optimize_frozen_transfer.py \
    --kinetics-data private_data/oxidation_dataset.xlsx \
    --pretrain-run outputs/pretraining \
    --output-dir outputs/optimization/frozen_transfer \
    --n-trials 100
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from pretrain_mechanism_encoder import (
    FEATURE_COLS,
    resolve_device,
)
from train_frozen_transfer import (
    KineticsDataset,
    FrozenTransferKineticsModel,
    evaluate_loader,
    load_kinetics_data,
    load_pretrained_backbone,
    set_seed,
    train_one_epoch,
)


# =============================================================================
# Fixed search space
# =============================================================================

WIDTH_CHOICES: Tuple[int, ...] = (
    32, 64, 128, 160, 192, 224, 256, 288, 320
)
LAYER_CHOICES: Tuple[int, ...] = (3, 4, 5)

DROPOUT_MIN = 0.00
DROPOUT_MAX = 0.30
DROPOUT_STEP = 0.05

LEARNING_RATE_MIN = 1e-4
LEARNING_RATE_MAX = 3e-3

WEIGHT_DECAY_MIN = 1e-6
WEIGHT_DECAY_MAX = 1e-3

BATCH_SIZE = 32

DEFAULT_EPOCHS = 3000
DEFAULT_PATIENCE = 200
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_TRIALS = 100

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 60
SCHEDULER_MIN_LR = 1e-7
GRADIENT_CLIP_NORM = 5.0

PRUNER_STARTUP_TRIALS = 10
PRUNER_WARMUP_EPOCHS = 30
PRUNER_INTERVAL_EPOCHS = 5


# =============================================================================
# Utilities
# =============================================================================

def save_json(payload: Mapping[str, Any], path: Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
            return None if not math.isfinite(value) else value
        if isinstance(value, float):
            return None if not math.isfinite(value) else value
        if isinstance(value, Path):
            return str(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            convert(dict(payload)),
            file,
            ensure_ascii=False,
            indent=2,
        )


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_architecture_choices() -> Tuple[str, ...]:
    """
    Enumerate strictly decreasing 3-5-layer kinetics-head architectures.
    """
    architectures: List[str] = []
    descending = tuple(sorted(WIDTH_CHOICES, reverse=True))

    for n_layers in LAYER_CHOICES:
        for widths in itertools.combinations(
            descending,
            n_layers,
        ):
            architectures.append(
                "-".join(str(width) for width in widths)
            )

    return tuple(architectures)


ARCHITECTURE_CHOICES = build_architecture_choices()


def parse_architecture(text: str) -> Tuple[int, ...]:
    dims = tuple(int(value) for value in text.split("-"))
    if len(dims) not in LAYER_CHOICES:
        raise ValueError(f"Invalid architecture: {text}")
    if any(
        dims[index] <= dims[index + 1]
        for index in range(len(dims) - 1)
    ):
        raise ValueError(
            "Frozen-transfer hidden widths must be strictly decreasing."
        )
    return dims


def prepare_development_data(
    kinetics_data: Path,
    sheet_name: str,
    pretrain_run: Path,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
]:
    """
    Load the fixed split and prepare TRAIN / VALIDATION only.

    TEST rows remain recorded in the source dataset but are removed before
    feature transformation, DataLoader creation, or model inference.
    """
    split_csv = pretrain_run / "data_split.csv"

    df_all = load_kinetics_data(
        path=kinetics_data,
        sheet_name=sheet_name,
        split_csv=split_csv,
    )

    split_counts = (
        df_all["split"]
        .value_counts()
        .to_dict()
    )

    dev = (
        df_all.loc[
            df_all["split"].isin(["train", "val"])
        ]
        .copy()
        .reset_index(drop=True)
    )

    train_idx = np.flatnonzero(
        dev["split"].eq("train").to_numpy()
    )
    val_idx = np.flatnonzero(
        dev["split"].eq("val").to_numpy()
    )

    scaler_path = pretrain_run / "feature_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Pretraining feature scaler not found: {scaler_path}"
        )

    scaler = joblib.load(scaler_path)

    x_raw = dev[list(FEATURE_COLS)].to_numpy(
        dtype=np.float32
    )
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    y = pd.to_numeric(
        dev["lgkp"],
        errors="raise",
    ).to_numpy(dtype=np.float32)

    return (
        dev,
        x_scaled,
        y,
        train_idx,
        val_idx,
        split_counts,
    )


def make_dataloaders(
    x_scaled: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
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

    common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **common,
    )

    return train_loader, val_loader


# =============================================================================
# Optuna objective
# =============================================================================

class FrozenTransferObjective:
    def __init__(
        self,
        x_scaled: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        pretrain_run: Path,
        device: torch.device,
        output_dir: Path,
        epochs: int,
        patience: int,
        random_seed: int,
        num_workers: int,
        log_interval: int,
    ) -> None:
        self.x_scaled = x_scaled
        self.y = y
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.pretrain_run = pretrain_run
        self.device = device
        self.output_dir = output_dir
        self.epochs = epochs
        self.patience = patience
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.log_interval = log_interval

    def __call__(self, trial: optuna.Trial) -> float:
        set_seed(self.random_seed)
        clear_device_cache(self.device)

        architecture_text = trial.suggest_categorical(
            "hidden_dims",
            ARCHITECTURE_CHOICES,
        )
        hidden_dims = parse_architecture(
            architecture_text
        )

        dropout = trial.suggest_float(
            "dropout",
            DROPOUT_MIN,
            DROPOUT_MAX,
            step=DROPOUT_STEP,
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            LEARNING_RATE_MIN,
            LEARNING_RATE_MAX,
            log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            WEIGHT_DECAY_MIN,
            WEIGHT_DECAY_MAX,
            log=True,
        )

        trial_dir = (
            self.output_dir
            / "trials"
            / f"trial_{trial.number:04d}"
        )
        trial_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader = make_dataloaders(
            x_scaled=self.x_scaled,
            y=self.y,
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            device=self.device,
            num_workers=self.num_workers,
        )

        # Reconstruct a fresh, frozen copy for every trial.
        backbone = load_pretrained_backbone(
            pretrain_run=self.pretrain_run,
            device=self.device,
        )

        model = FrozenTransferKineticsModel(
            backbone=backbone,
            kinetics_hidden_dims=hidden_dims,
            kinetics_dropout=dropout,
        ).to(self.device)

        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if any(
            not name.startswith("kinetics_head.")
            for name in trainable
        ):
            raise RuntimeError(
                "Frozen-transfer safety check failed: "
                "a pretrained parameter is trainable."
            )

        optimizer = torch.optim.AdamW(
            model.kinetics_head.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=SCHEDULER_FACTOR,
                patience=SCHEDULER_PATIENCE,
                min_lr=SCHEDULER_MIN_LR,
            )
        )

        best_rmse = float("inf")
        best_epoch = -1
        best_metrics: Dict[str, float] = {}
        wait = 0
        history: List[Dict[str, Any]] = []

        try:
            for epoch in range(1, self.epochs + 1):
                train_mse = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=self.device,
                    gradient_clip_norm=GRADIENT_CLIP_NORM,
                )

                validation, _ = evaluate_loader(
                    model=model,
                    loader=val_loader,
                    device=self.device,
                )

                val_rmse = float(validation["RMSE"])
                if not np.isfinite(val_rmse):
                    raise RuntimeError(
                        "Validation RMSE became non-finite."
                    )

                scheduler.step(val_rmse)

                history.append(
                    {
                        "epoch": epoch,
                        "train_MSE": train_mse,
                        "val_MSE": float(validation["MSE"]),
                        "val_R2": float(validation["R2"]),
                        "val_MAE": float(validation["MAE"]),
                        "val_RMSE": val_rmse,
                        "learning_rate": float(
                            optimizer.param_groups[0]["lr"]
                        ),
                    }
                )

                trial.report(val_rmse, step=epoch)

                if trial.should_prune():
                    pd.DataFrame(history).to_csv(
                        trial_dir / "history.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch}."
                    )

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_epoch = epoch
                    best_metrics = dict(validation)
                    wait = 0
                else:
                    wait += 1

                if (
                    epoch == 1
                    or epoch % self.log_interval == 0
                ):
                    print(
                        f"Trial {trial.number:04d} | "
                        f"epoch={epoch:04d} | "
                        f"val RMSE={val_rmse:.5f} | "
                        f"best={best_rmse:.5f} | "
                        f"wait={wait}/{self.patience}",
                        flush=True,
                    )

                if wait >= self.patience:
                    break

            if best_epoch < 0:
                raise RuntimeError(
                    "No valid frozen-transfer epoch was produced."
                )

            pd.DataFrame(history).to_csv(
                trial_dir / "history.csv",
                index=False,
                encoding="utf-8-sig",
            )

            save_json(
                {
                    "trial_number": trial.number,
                    "objective": "validation_RMSE",
                    "best_epoch": best_epoch,
                    "best_validation_RMSE": best_rmse,
                    "hidden_dims": list(hidden_dims),
                    "dropout": dropout,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "validation_metrics_at_best_epoch":
                        best_metrics,
                    "pretrained_backbone_frozen": True,
                    "training_objective": "unweighted_MSE",
                    "test_used": False,
                },
                trial_dir / "trial_summary.json",
            )

            trial.set_user_attr(
                "best_epoch",
                best_epoch,
            )
            trial.set_user_attr(
                "validation_R2",
                best_metrics.get("R2"),
            )
            trial.set_user_attr(
                "validation_MAE",
                best_metrics.get("MAE"),
            )

            return best_rmse

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                trial.set_user_attr(
                    "failure",
                    "cuda_out_of_memory",
                )
                raise optuna.TrialPruned(
                    "CUDA out of memory."
                ) from exc
            raise

        finally:
            del model
            del optimizer
            clear_device_cache(self.device)


# =============================================================================
# Study export
# =============================================================================

def export_best_config(
    study: optuna.Study,
    output_dir: Path,
    epochs: int,
    patience: int,
    random_seed: int,
    num_workers: int,
    log_interval: int,
) -> None:
    best = study.best_trial
    hidden_dims = parse_architecture(
        str(best.params["hidden_dims"])
    )

    # Directly compatible with train_frozen_transfer.py.
    config = {
        "hidden_dims": list(hidden_dims),
        "dropout": float(best.params["dropout"]),
        "batch_size": BATCH_SIZE,
        "epochs": int(epochs),
        "learning_rate": float(
            best.params["learning_rate"]
        ),
        "weight_decay": float(
            best.params["weight_decay"]
        ),
        "patience": int(patience),
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "random_seed": int(random_seed),
        "num_workers": int(num_workers),
        "log_interval": int(log_interval),
    }

    save_json(
        config,
        output_dir / "best_frozen_transfer_config.json",
    )

    save_json(
        {
            "study_name": study.study_name,
            "direction": "minimize",
            "objective": "validation_RMSE",
            "best_trial_number": best.number,
            "best_value": best.value,
            "best_params": dict(best.params),
            "best_user_attrs": dict(best.user_attrs),
            "test_used_for_selection": False,
        },
        output_dir / "best_trial.json",
    )


def export_search_space(output_dir: Path) -> None:
    save_json(
        {
            "transfer_mode": "frozen_only",
            "searched_hyperparameters": [
                "kinetics_head_depth",
                "kinetics_head_hidden_width",
                "dropout",
                "learning_rate",
                "weight_decay",
            ],
            "architecture": {
                "layers": list(LAYER_CHOICES),
                "width_choices": list(WIDTH_CHOICES),
                "ordering": "strictly_decreasing",
                "number_of_architectures":
                    len(ARCHITECTURE_CHOICES),
            },
            "dropout": {
                "min": DROPOUT_MIN,
                "max": DROPOUT_MAX,
                "step": DROPOUT_STEP,
            },
            "learning_rate": {
                "min": LEARNING_RATE_MIN,
                "max": LEARNING_RATE_MAX,
                "log": True,
            },
            "weight_decay": {
                "min": WEIGHT_DECAY_MIN,
                "max": WEIGHT_DECAY_MAX,
                "log": True,
            },
            "batch_size": {
                "value": BATCH_SIZE,
                "searched": False,
            },
            "scheduler": {
                "factor": SCHEDULER_FACTOR,
                "patience": SCHEDULER_PATIENCE,
                "minimum_lr": SCHEDULER_MIN_LR,
            },
            "training_objective": "unweighted_MSE",
            "optuna_objective": "validation_RMSE",
            "pretrained_backbone": "frozen",
            "mechanism_alignment_loss": False,
            "auxiliary_classification_loss": False,
            "quality_weighting": False,
            "test_used": False,
        },
        output_dir / "search_space.json",
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna optimization for the paper-release "
            "frozen-transfer kinetics model."
        )
    )

    parser.add_argument(
        "--kinetics-data",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--sheet-name",
        default="Sheet1",
    )
    parser.add_argument(
        "--pretrain-run",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/optimization/frozen_transfer"
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trials").mkdir(
        parents=True,
        exist_ok=True,
    )

    pretrain_run = args.pretrain_run.resolve()
    if not pretrain_run.exists():
        raise FileNotFoundError(
            f"Pretraining run directory not found: {pretrain_run}"
        )

    device = resolve_device(args.device)
    set_seed(args.random_seed)

    (
        dev_df,
        x_scaled,
        y,
        train_idx,
        val_idx,
        split_counts,
    ) = prepare_development_data(
        kinetics_data=args.kinetics_data,
        sheet_name=args.sheet_name,
        pretrain_run=pretrain_run,
    )

    save_json(
        {
            "full_fixed_split_counts": split_counts,
            "development_rows_loaded": int(len(dev_df)),
            "development_uses": [
                "train",
                "val",
            ],
            "test_rows_transformed": False,
            "test_rows_inferred": False,
            "test_used_for_selection": False,
        },
        output_dir / "data_usage.json",
    )

    export_search_space(output_dir)

    storage_path = output_dir / "optuna_study.db"
    storage_url = (
        "sqlite:///" + storage_path.resolve().as_posix()
    )

    sampler = TPESampler(
        seed=args.random_seed,
        multivariate=True,
        group=True,
    )
    pruner = MedianPruner(
        n_startup_trials=PRUNER_STARTUP_TRIALS,
        n_warmup_steps=PRUNER_WARMUP_EPOCHS,
        interval_steps=PRUNER_INTERVAL_EPOCHS,
    )

    study = optuna.create_study(
        study_name="frozen_transfer",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=bool(args.resume),
    )

    print("=" * 96)
    print("Optuna frozen-transfer optimization")
    print("=" * 96)
    print(f"Trials:              {args.n_trials}")
    print("Objective:           validation RMSE")
    print(f"Batch size:          {BATCH_SIZE} (fixed)")
    print(f"Architectures:       {len(ARCHITECTURE_CHOICES)}")
    print("Backbone:            frozen")
    print("Test used:           NO")
    print(f"Device:              {device}")
    print(f"Output:              {output_dir}")
    print("=" * 96)

    objective = FrozenTransferObjective(
        x_scaled=x_scaled,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        pretrain_run=pretrain_run,
        device=device,
        output_dir=output_dir,
        epochs=args.epochs,
        patience=args.patience,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "params",
            "state",
            "user_attrs",
        )
    ).to_csv(
        output_dir / "study_trials.csv",
        index=False,
        encoding="utf-8-sig",
    )

    export_best_config(
        study=study,
        output_dir=output_dir,
        epochs=args.epochs,
        patience=args.patience,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
    )

    print("-" * 96)
    print(
        f"Best trial:          {study.best_trial.number}"
    )
    print(
        f"Best validation RMSE:{study.best_value:.6f}"
    )
    print(
        f"Best params:         {study.best_params}"
    )
    print(
        "Final config:        "
        f"{output_dir / 'best_frozen_transfer_config.json'}"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
