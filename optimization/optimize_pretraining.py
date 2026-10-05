#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optuna hyperparameter optimization for six-channel mechanism pretraining.

Paper-release implementation
----------------------------
This script searches only the hyperparameters described in the manuscript:

- network depth
- hidden-layer width
- dropout
- learning rate
- weight decay

Fixed choices
-------------
- six mechanism channels
- 64-D mechanism targets
- cosine-distance supervision only
- batch size = 32
- one fixed train / validation / test split
- feature scaler fitted on TRAIN only by pretrain_mechanism_encoder.prepare_data
- Optuna objective = mean validation cosine similarity across the six channels
- TEST data are never used for trial ranking, pruning, early stopping, or
  hyperparameter selection

The best hyperparameter set is exported as a JSON file directly compatible with
src/pretrain_mechanism_encoder.py.

Typical use
-----------
python optimization/optimize_pretraining.py \
    --structured-data private_data/oxidation_dataset.xlsx \
    --reference-root outputs/mechanism_embeddings \
    --split-csv private_data/split_manifest.csv \
    --output-dir outputs/optimization/pretraining \
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

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from pretrain_mechanism_encoder import (
    FEATURE_COLS,
    SELECTED_CHANNELS,
    MechanismDataset,
    MultiChannelMechanismEncoder,
    compute_multichannel_loss,
    evaluate_validation,
    prepare_data,
    resolve_device,
    set_seed,
)


# =============================================================================
# Fixed search space used for the paper-release workflow
# =============================================================================

WIDTH_CHOICES: Tuple[int, ...] = (
    64, 96, 128, 160, 192, 224, 256, 288, 320
)
LAYER_CHOICES: Tuple[int, ...] = (3, 4, 5)

DROPOUT_MIN = 0.05
DROPOUT_MAX = 0.15
DROPOUT_STEP = 0.05

LEARNING_RATE_MIN = 1e-4
LEARNING_RATE_MAX = 1e-3

WEIGHT_DECAY_MIN = 1e-7
WEIGHT_DECAY_MAX = 1e-3

BATCH_SIZE = 32

DEFAULT_EPOCHS = 4000
DEFAULT_PATIENCE = 200
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_TRIALS = 100

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
    Enumerate non-increasing 3-5-layer architectures.

    A fixed categorical architecture list avoids conditional Optuna parameters
    and keeps the search definition transparent in the released code.
    """
    architectures: List[str] = []
    descending = tuple(sorted(WIDTH_CHOICES, reverse=True))

    for n_layers in LAYER_CHOICES:
        for widths in itertools.combinations_with_replacement(
            descending,
            n_layers,
        ):
            if all(
                widths[index] >= widths[index + 1]
                for index in range(len(widths) - 1)
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
    return dims


def make_dataloaders(
    prepared: Any,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_idx = np.flatnonzero(
        prepared.df["split"].eq("train").to_numpy()
    )
    val_idx = np.flatnonzero(
        prepared.df["split"].eq("val").to_numpy()
    )

    train_ds = MechanismDataset(
        x=prepared.x_scaled,
        row_indices=train_idx,
        channel_targets=prepared.channel_targets,
        channel_masks=prepared.channel_masks,
    )
    val_ds = MechanismDataset(
        x=prepared.x_scaled,
        row_indices=val_idx,
        channel_targets=prepared.channel_targets,
        channel_masks=prepared.channel_masks,
    )

    common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    return train_loader, val_loader


# =============================================================================
# Optuna objective
# =============================================================================

class PretrainingObjective:
    def __init__(
        self,
        prepared: Any,
        device: torch.device,
        output_dir: Path,
        epochs: int,
        patience: int,
        random_seed: int,
        num_workers: int,
        log_interval: int,
    ) -> None:
        self.prepared = prepared
        self.device = device
        self.output_dir = output_dir
        self.epochs = epochs
        self.patience = patience
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.log_interval = log_interval

    def __call__(self, trial: optuna.Trial) -> float:
        # The same initialization seed is used across trials so architecture /
        # optimizer comparisons are not confounded by a changing data split.
        set_seed(self.random_seed)
        clear_device_cache(self.device)

        architecture_text = trial.suggest_categorical(
            "hidden_dims",
            ARCHITECTURE_CHOICES,
        )
        hidden_dims = parse_architecture(architecture_text)

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
            prepared=self.prepared,
            batch_size=BATCH_SIZE,
            device=self.device,
            num_workers=self.num_workers,
        )

        model = MultiChannelMechanismEncoder(
            in_dim=len(FEATURE_COLS),
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_score = -float("inf")
        best_epoch = -1
        best_metrics: Dict[str, float] = {}
        wait = 0
        history: List[Dict[str, Any]] = []

        try:
            for epoch in range(1, self.epochs + 1):
                model.train()
                train_losses: List[float] = []
                skipped_batches = 0

                for batch in train_loader:
                    x = batch["x"].to(
                        self.device,
                        non_blocking=True,
                    )

                    # Shared encoder uses BatchNorm.
                    if x.shape[0] < 2:
                        skipped_batches += 1
                        continue

                    predictions = model(x)
                    loss, _ = compute_multichannel_loss(
                        predictions=predictions,
                        batch=batch,
                        device=self.device,
                    )

                    if loss is None:
                        skipped_batches += 1
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(
                        float(loss.detach().cpu())
                    )

                validation = evaluate_validation(
                    model=model,
                    loader=val_loader,
                    device=self.device,
                )

                # Manuscript-defined hyperparameter-selection objective.
                score = float(
                    validation["mean_channel_cosine"]
                )
                if not np.isfinite(score):
                    raise RuntimeError(
                        "Mean validation cosine similarity became non-finite."
                    )

                row: Dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": (
                        float(np.mean(train_losses))
                        if train_losses
                        else float("nan")
                    ),
                    "skipped_train_batches": skipped_batches,
                    **validation,
                }
                history.append(row)

                trial.report(score, step=epoch)

                if trial.should_prune():
                    pd.DataFrame(history).to_csv(
                        trial_dir / "history.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch}."
                    )

                if score > best_score:
                    best_score = score
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
                        f"mean val CS={score:.5f} | "
                        f"best={best_score:.5f} | "
                        f"wait={wait}/{self.patience}",
                        flush=True,
                    )

                if wait >= self.patience:
                    break

            if best_epoch < 0:
                raise RuntimeError(
                    "No valid pretraining epoch was produced."
                )

            pd.DataFrame(history).to_csv(
                trial_dir / "history.csv",
                index=False,
                encoding="utf-8-sig",
            )

            summary = {
                "trial_number": trial.number,
                "objective": (
                    "mean validation cosine similarity "
                    "across six channels"
                ),
                "best_epoch": best_epoch,
                "best_objective_value": best_score,
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
                "batch_size": BATCH_SIZE,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "validation_metrics_at_best_epoch":
                    best_metrics,
                "test_used": False,
            }
            save_json(
                summary,
                trial_dir / "trial_summary.json",
            )

            trial.set_user_attr(
                "best_epoch",
                best_epoch,
            )
            trial.set_user_attr(
                "hidden_dims",
                list(hidden_dims),
            )

            for channel in SELECTED_CHANNELS:
                key = f"ch{channel}_cosine_mean"
                if key in best_metrics:
                    trial.set_user_attr(
                        f"validation_{key}",
                        best_metrics[key],
                    )

            return best_score

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

    # Directly compatible with pretrain_mechanism_encoder.py.
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
        "random_seed": int(random_seed),
        "num_workers": int(num_workers),
        "log_interval": int(log_interval),
    }

    save_json(
        config,
        output_dir / "best_pretraining_config.json",
    )

    save_json(
        {
            "study_name": study.study_name,
            "direction": "maximize",
            "objective": (
                "mean validation cosine similarity "
                "across six channels"
            ),
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
            "searched_hyperparameters": [
                "network_depth",
                "hidden_layer_width",
                "dropout",
                "learning_rate",
                "weight_decay",
            ],
            "architecture": {
                "layers": list(LAYER_CHOICES),
                "width_choices": list(WIDTH_CHOICES),
                "ordering": "non-increasing",
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
            "objective": (
                "mean validation cosine similarity "
                "across six mechanism channels"
            ),
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
            "six-channel mechanism-pretraining model."
        )
    )

    parser.add_argument(
        "--structured-data",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--sheet-name",
        default="Sheet3",
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--split-csv",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/optimization/pretraining"
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
        help=(
            "Resume the SQLite Optuna study in output-dir."
        ),
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

    device = resolve_device(args.device)
    set_seed(args.random_seed)

    # Prepared exactly once. This preserves the same split and TRAIN-fitted
    # scaler for every Optuna trial.
    prepared = prepare_data(
        structured_data=args.structured_data,
        sheet_name=args.sheet_name,
        reference_root=args.reference_root,
        split_csv=args.split_csv,
    )

    split_counts = (
        prepared.df["split"]
        .value_counts()
        .to_dict()
    )

    # Explicitly document that test rows exist in the fixed split but are not
    # passed to any Optuna DataLoader or objective computation.
    save_json(
        {
            "split_counts": split_counts,
            "optimization_uses": [
                "train",
                "val",
            ],
            "test_used": False,
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
        study_name="mechanism_pretraining",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=bool(args.resume),
    )

    print("=" * 96)
    print("Optuna mechanism-pretraining optimization")
    print("=" * 96)
    print(f"Trials:              {args.n_trials}")
    print(f"Objective:           mean validation CS")
    print(f"Batch size:          {BATCH_SIZE} (fixed)")
    print(f"Architectures:       {len(ARCHITECTURE_CHOICES)}")
    print(f"Test used:           NO")
    print(f"Device:              {device}")
    print(f"Output:              {output_dir}")
    print("=" * 96)

    objective = PretrainingObjective(
        prepared=prepared,
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
        f"Best trial:     {study.best_trial.number}"
    )
    print(
        f"Best mean CS:   {study.best_value:.6f}"
    )
    print(
        f"Best params:    {study.best_params}"
    )
    print(
        "Final config:   "
        f"{output_dir / 'best_pretraining_config.json'}"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
