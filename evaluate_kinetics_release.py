#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final held-out evaluation of the frozen-transfer oxidation-kinetics model.

Paper-release implementation
----------------------------
This script performs the first and only final evaluation of the selected
frozen-transfer checkpoint on the held-out TEST subset.

It reports:
    - R2
    - MAE
    - RMSE

and exports:
    - held-out experimental/predicted lg(kp) values (Fig. 5b source data)
    - the six predicted 64-D mechanism vectors for each held-out sample,
      which can be reused by the downstream mechanism-interpretation scripts
    - a parity plot for rapid reproduction/checking of Fig. 5(b)

No fitting, checkpoint selection, hyperparameter optimization, or test-guided
model choice is performed here.

Expected transfer run directory
-------------------------------
<transfer_run>/
    best_transfer_model_state.pt
    best_transfer_checkpoint.pt
    resolved_transfer_config.json
    data_split.csv

Expected pretraining run directory
----------------------------------
<pretrain_run>/
    best_model_state.pt
    feature_scaler.joblib
    resolved_pretraining_config.json
    data_split.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from pretrain_mechanism_encoder import (
    ALIGN_KEYS,
    CHANNEL_NAMES,
    EMBED_DIM,
    FEATURE_COLS,
    SELECTED_CHANNELS,
    MultiChannelMechanismEncoder,
    resolve_device,
)
from train_frozen_transfer import (
    FrozenTransferKineticsModel,
    KineticsDataset,
    load_kinetics_data,
)


TARGET_COL = "lgkp"


# =============================================================================
# Generic helpers
# =============================================================================


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return payload


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty test set.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Target/prediction length mismatch: {len(y_true)} vs {len(y_pred)}"
        )

    return {
        "n": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# =============================================================================
# Model reconstruction
# =============================================================================


def load_pretrained_backbone(
    pretrain_run: Path,
    device: torch.device,
) -> MultiChannelMechanismEncoder:
    """Reconstruct the mechanism-pretrained backbone exactly."""

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
        raise FileNotFoundError(
            f"Pretraining checkpoint not found: {checkpoint}"
        )

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    backbone.load_state_dict(state_dict, strict=True)

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    backbone.to(device)
    backbone.eval()

    return backbone


def load_transfer_model(
    transfer_run: Path,
    pretrain_run: Path,
    device: torch.device,
) -> FrozenTransferKineticsModel:
    """Reconstruct the selected frozen-transfer model and load its checkpoint."""

    transfer_config = read_json(
        transfer_run / "resolved_transfer_config.json"
    )

    hidden_dims = tuple(
        int(x) for x in transfer_config["hidden_dims"]
    )
    dropout = float(transfer_config["dropout"])

    backbone = load_pretrained_backbone(
        pretrain_run=pretrain_run,
        device=device,
    )

    model = FrozenTransferKineticsModel(
        backbone=backbone,
        kinetics_hidden_dims=hidden_dims,
        kinetics_dropout=dropout,
    ).to(device)

    checkpoint = transfer_run / "best_transfer_model_state.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Frozen-transfer checkpoint not found: {checkpoint}"
        )

    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Safety check: the loaded release model must remain a frozen-backbone model.
    for name, parameter in model.backbone.named_parameters():
        if parameter.requires_grad:
            raise RuntimeError(
                f"Backbone parameter unexpectedly remains trainable: {name}"
            )

    return model


# =============================================================================
# Held-out inference
# =============================================================================


@torch.no_grad()
def predict_test_set(
    model: FrozenTransferKineticsModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[int, np.ndarray],
]:
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []
    global_index_parts: List[np.ndarray] = []
    mechanism_parts: Dict[int, List[np.ndarray]] = {
        channel: [] for channel in SELECTED_CHANNELS
    }

    model.eval()

    for batch in loader:
        x = batch["x"].to(device)
        outputs = model(x)

        y_true_parts.append(
            batch["y"].cpu().numpy()
        )
        y_pred_parts.append(
            outputs["lgkp"].cpu().numpy()
        )
        global_index_parts.append(
            batch["global_index"].cpu().numpy()
        )

        for channel in SELECTED_CHANNELS:
            mechanism_parts[channel].append(
                outputs["mechanism_vectors"][str(channel)]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    y_true = np.concatenate(y_true_parts).astype(np.float64)
    y_pred = np.concatenate(y_pred_parts).astype(np.float64)
    global_indices = np.concatenate(global_index_parts).astype(np.int64)

    mechanism_vectors = {
        channel: np.vstack(parts).astype(np.float32)
        for channel, parts in mechanism_parts.items()
    }

    return y_true, y_pred, global_indices, mechanism_vectors


# =============================================================================
# Publication/source-data exports
# =============================================================================


def build_test_prediction_frame(
    df: pd.DataFrame,
    global_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    keep_columns = [
        *ALIGN_KEYS,
        *FEATURE_COLS,
        "split",
    ]

    available_columns = [
        column for column in keep_columns
        if column in df.columns
    ]

    out = (
        df.iloc[global_indices][available_columns]
        .copy()
        .reset_index(drop=True)
    )

    out["lgkp_true"] = y_true
    out["lgkp_pred"] = y_pred
    out["residual"] = y_pred - y_true
    out["abs_error"] = np.abs(out["residual"])

    return out


def build_mechanism_vector_frame(
    df: pd.DataFrame,
    global_indices: np.ndarray,
    mechanism_vectors: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    """
    Export one row per held-out sample x mechanism channel.

    These are model-predicted representations, not literature-reference vectors.
    """
    rows: List[pd.DataFrame] = []

    identity = (
        df.iloc[global_indices][list(ALIGN_KEYS)]
        .copy()
        .reset_index(drop=True)
    )

    for channel in SELECTED_CHANNELS:
        array = np.asarray(
            mechanism_vectors[channel],
            dtype=np.float32,
        )

        if len(array) != len(identity):
            raise RuntimeError(
                f"Channel {channel} vector row mismatch: "
                f"{len(array)} vs {len(identity)}"
            )

        part = identity.copy()
        part["channel"] = channel
        part["channel_name"] = CHANNEL_NAMES[channel]

        vector_df = pd.DataFrame(
            array,
            columns=[
                f"emb_{index}"
                for index in range(EMBED_DIM)
            ],
        )

        part = pd.concat(
            [part, vector_df],
            axis=1,
        )
        rows.append(part)

    return pd.concat(rows, ignore_index=True)


def plot_test_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Mapping[str, float],
    output_dir: Path,
) -> None:
    """Simple parity plot underlying Fig. 5(b)."""

    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    pad = max((upper - lower) * 0.06, 0.10)
    lower -= pad
    upper += pad

    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    ax.scatter(
        y_true,
        y_pred,
        s=30,
        alpha=0.75,
        edgecolors="none",
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"Experimental lg$k_p$")
    ax.set_ylabel(r"Predicted lg$k_p$")

    annotation = (
        f"$R^2$ = {metrics['R2']:.4f}\n"
        f"MAE = {metrics['MAE']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.4f}"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "fig5b_test_parity.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "fig5b_test_parity.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the selected frozen-transfer checkpoint "
            "once on the held-out test subset."
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
        "--transfer-run",
        required=True,
        type=Path,
        help="Final frozen-transfer run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pretrain_run = args.pretrain_run.resolve()
    transfer_run = args.transfer_run.resolve()

    if not pretrain_run.exists():
        raise FileNotFoundError(
            f"Pretraining run directory not found: {pretrain_run}"
        )
    if not transfer_run.exists():
        raise FileNotFoundError(
            f"Transfer run directory not found: {transfer_run}"
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else transfer_run / "test_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    # ------------------------------------------------------------------
    # Fixed split identity is inherited from mechanism pretraining.
    # ------------------------------------------------------------------
    df = load_kinetics_data(
        path=args.kinetics_data,
        sheet_name=args.sheet_name,
        split_csv=pretrain_run / "data_split.csv",
    )

    test_idx = np.flatnonzero(
        df["split"].eq("test").to_numpy(dtype=bool)
    )

    if len(test_idx) == 0:
        raise ValueError("No held-out kinetics test records were found.")

    # ------------------------------------------------------------------
    # Reuse the exact feature scaler learned during mechanism pretraining.
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

    test_ds = KineticsDataset(
        x_scaled=x_scaled,
        y=y,
        row_indices=test_idx,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = load_transfer_model(
        transfer_run=transfer_run,
        pretrain_run=pretrain_run,
        device=device,
    )

    y_true, y_pred, global_indices, mechanism_vectors = predict_test_set(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics = regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    # ------------------------------------------------------------------
    # Held-out source data
    # ------------------------------------------------------------------
    prediction_df = build_test_prediction_frame(
        df=df,
        global_indices=global_indices,
        y_true=y_true,
        y_pred=y_pred,
    )
    prediction_df.to_csv(
        output_dir / "fig5b_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mechanism_df = build_mechanism_vector_frame(
        df=df,
        global_indices=global_indices,
        mechanism_vectors=mechanism_vectors,
    )
    mechanism_df.to_csv(
        output_dir / "test_predicted_mechanism_vectors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Compact array version for downstream case-analysis code.
    np.savez_compressed(
        output_dir / "test_predicted_mechanism_vectors.npz",
        **{
            f"channel_{channel}": mechanism_vectors[channel]
            for channel in SELECTED_CHANNELS
        },
        global_indices=global_indices,
    )

    with (
        output_dir / "test_kinetics_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "evaluation_split": "test",
                "checkpoint_selected_without_test_data": True,
                "n": metrics["n"],
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(
        [
            {
                "split": "test",
                **metrics,
            }
        ]
    ).to_csv(
        output_dir / "test_kinetics_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_test_parity(
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
        output_dir=output_dir,
    )

    print("=" * 88)
    print("Held-out frozen-transfer kinetics evaluation")
    print("=" * 88)
    print(f"Test records: {metrics['n']}")
    print(f"R2:           {metrics['R2']:.6f}")
    print(f"MAE:          {metrics['MAE']:.6f}")
    print(f"RMSE:         {metrics['RMSE']:.6f}")
    print(f"Outputs:      {output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
