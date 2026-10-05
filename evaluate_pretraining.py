# -*- coding: utf-8 -*-

"""
Evaluate the final six-channel mechanism-pretraining model on the held-out test set.

Paper-release implementation
----------------------------
This script reproduces the three evaluation levels reported in the manuscript:

1. Pointwise representation alignment
   - cosine similarity (CS) between each predicted mechanism representation and
     its corresponding literature-derived reference representation.
   - channel-wise mean, median, and 25th percentile CS.

2. Pairwise similarity preservation (PSP)
   - cosine similarities are computed for all valid test-set sample pairs
     separately in the reference and predicted mechanism spaces.
   - Spearman's rank correlation coefficient (rho) quantifies preservation of
     the inter-state similarity structure.

3. Reference-state retrieval
   - each predicted test representation is used as a query against the valid
     held-out reference representations from the same mechanism channel.
   - the rank of the corresponding reference state is recorded.
   - MRR, median rank, Recall@3, and Recall@5 are reported.

No model fitting, hyperparameter optimization, or UMAP fitting is performed here.
The test split is used only for final evaluation.

Expected pretraining run directory
----------------------------------
<run_dir>/
    best_model_state.pt
    feature_scaler.joblib
    resolved_pretraining_config.json
    data_split.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from pretrain_mechanism_encoder import (
    ALIGN_KEYS,
    CHANNEL_NAMES,
    EMBED_DIM,
    FEATURE_COLS,
    SELECTED_CHANNELS,
    MultiChannelMechanismEncoder,
    attach_fixed_split,
    attach_reference_spaces,
    load_split_manifest,
    load_structured_data,
    resolve_device,
)


# =============================================================================
# Helpers
# =============================================================================

SHORT_CHANNEL_NAMES: Dict[int, str] = {
    1: "Protective scale",
    2: "Internal/selective oxidation",
    3: "Non-protective/transient oxides",
    4: "Transport/defect kinetics",
    5: "Scale failure/spallation",
    6: "Alloying-element interaction",
}


def read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, eps, None)


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return np.sum(l2_normalize(a) * l2_normalize(b), axis=1)


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    a_norm = l2_normalize(a)
    if b is None:
        b_norm = a_norm
    else:
        b_norm = l2_normalize(b)
    return a_norm @ b_norm.T


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    result = spearmanr(x, y, nan_policy="omit")
    return float(result.statistic)


# =============================================================================
# Data preparation
# =============================================================================

def prepare_test_data(
    structured_data: Path,
    sheet_name: str,
    reference_root: Path,
    split_csv: Path,
    scaler_path: Path,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Align structured inputs, six reference spaces, and the saved split identity.

    The feature scaler is loaded from the pretraining run and is not refitted.
    """
    df = load_structured_data(structured_data, sheet_name)
    df = attach_reference_spaces(df, reference_root)
    df = attach_fixed_split(df, load_split_manifest(split_csv))

    scaler = joblib.load(scaler_path)

    x = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32)

    targets: Dict[int, np.ndarray] = {}
    masks: Dict[int, np.ndarray] = {}

    for channel in SELECTED_CHANNELS:
        emb_cols = [f"ch{channel}__emb_{i}" for i in range(EMBED_DIM)]
        targets[channel] = df[emb_cols].to_numpy(dtype=np.float32)

        valid = pd.to_numeric(
            df[f"ch{channel}__embedding_valid"],
            errors="raise",
        ).astype(int)

        if not valid.isin([0, 1]).all():
            raise ValueError(
                f"Invalid embedding_valid values in channel {channel}."
            )

        masks[channel] = valid.eq(1).to_numpy(dtype=bool)

    return df, x_scaled, targets, masks


# =============================================================================
# Model loading and inference
# =============================================================================

def load_model(
    run_dir: Path,
    device: torch.device,
) -> MultiChannelMechanismEncoder:
    config = read_json(run_dir / "resolved_pretraining_config.json")

    hidden_dims = tuple(int(x) for x in config["hidden_dims"])
    dropout = float(config["dropout"])

    model = MultiChannelMechanismEncoder(
        in_dim=len(FEATURE_COLS),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    checkpoint = run_dir / "best_model_state.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_all(
    model: MultiChannelMechanismEncoder,
    x_scaled: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    predictions: Dict[int, List[np.ndarray]] = {
        channel: [] for channel in SELECTED_CHANNELS
    }

    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            xb = torch.as_tensor(
                x_scaled[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            outputs = model(xb)

            for channel in SELECTED_CHANNELS:
                predictions[channel].append(
                    outputs[str(channel)].cpu().numpy()
                )

    return {
        channel: np.vstack(parts).astype(np.float32)
        for channel, parts in predictions.items()
    }


# =============================================================================
# Evaluation metrics
# =============================================================================

def reference_retrieval(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve each sample's corresponding held-out reference state.

    Query:
        predicted mechanism representation

    Candidate library:
        all valid held-out reference representations in the same channel

    The correct reference for query i is reference i.
    """
    similarity = cosine_matrix(prediction, reference)
    order = np.argsort(-similarity, axis=1)

    ranks = np.empty(len(reference), dtype=np.int64)

    for i in range(len(reference)):
        location = np.flatnonzero(order[i] == i)
        if len(location) != 1:
            raise RuntimeError(
                f"Could not uniquely identify the correct reference for row {i}."
            )
        ranks[i] = int(location[0]) + 1

    return ranks, similarity


def evaluate_channels(
    df: pd.DataFrame,
    predictions: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    masks: Mapping[int, np.ndarray],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Returns
    -------
    pointwise_summary
        Table-1-style channel-wise CS statistics.

    retrieval_summary
        Table-2-style reference-state retrieval metrics.

    samplewise
        Per-sample CS and reference rank.

    pairwise
        Pairwise reference/predicted similarities for Fig. 4.
    """
    test_mask = df["split"].eq("test").to_numpy(dtype=bool)

    pointwise_rows: List[Dict] = []
    retrieval_rows: List[Dict] = []
    samplewise_rows: List[Dict] = []
    pairwise_rows: List[Dict] = []

    for channel in SELECTED_CHANNELS:
        valid = test_mask & masks[channel]
        row_indices = np.flatnonzero(valid)

        reference = np.asarray(targets[channel][valid], dtype=np.float32)
        prediction = np.asarray(predictions[channel][valid], dtype=np.float32)

        if len(reference) < 2:
            raise ValueError(
                f"Channel {channel} has fewer than two valid test records."
            )

        # ------------------------------------------------------------------
        # 1. Pointwise CS
        # ------------------------------------------------------------------
        cs = row_cosine(prediction, reference)

        pointwise_rows.append(
            {
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "n_valid": int(len(reference)),
                "cosine_mean": float(np.mean(cs)),
                "cosine_median": float(np.median(cs)),
                "cosine_p25": float(np.percentile(cs, 25)),
            }
        )

        # ------------------------------------------------------------------
        # 2. Pairwise similarity preservation
        # ------------------------------------------------------------------
        reference_sim = cosine_matrix(reference)
        predicted_sim = cosine_matrix(prediction)

        ref_pairs = upper_triangle_values(reference_sim)
        pred_pairs = upper_triangle_values(predicted_sim)

        rho = safe_spearman(ref_pairs, pred_pairs)

        # Build explicit pair table for source-data reproduction.
        local_i, local_j = np.triu_indices(len(reference), k=1)

        for i_local, j_local, ref_value, pred_value in zip(
            local_i,
            local_j,
            ref_pairs,
            pred_pairs,
        ):
            i_global = int(row_indices[i_local])
            j_global = int(row_indices[j_local])

            pairwise_rows.append(
                {
                    "channel": channel,
                    "channel_name": CHANNEL_NAMES[channel],
                    "sample_id_i": df.iloc[i_global]["sample_id"],
                    "sample_id_j": df.iloc[j_global]["sample_id"],
                    "reference_similarity": float(ref_value),
                    "predicted_similarity": float(pred_value),
                    "spearman_rho_channel": rho,
                }
            )

        # ------------------------------------------------------------------
        # 3. Reference-state retrieval
        # ------------------------------------------------------------------
        ranks, similarity_matrix = reference_retrieval(
            prediction=prediction,
            reference=reference,
        )

        retrieval_rows.append(
            {
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "n_valid": int(len(reference)),
                "MRR": float(np.mean(1.0 / ranks)),
                "median_rank": float(np.median(ranks)),
                "Recall@3": float(np.mean(ranks <= 3)),
                "Recall@5": float(np.mean(ranks <= 5)),
            }
        )

        # ------------------------------------------------------------------
        # Per-sample source data
        # ------------------------------------------------------------------
        for local_index, global_index in enumerate(row_indices):
            row = df.iloc[int(global_index)]

            samplewise_rows.append(
                {
                    **{
                        key: row[key]
                        for key in ALIGN_KEYS
                    },
                    "channel": channel,
                    "channel_name": CHANNEL_NAMES[channel],
                    "CS": float(cs[local_index]),
                    "reference_rank": int(ranks[local_index]),
                    "self_reference_similarity": float(
                        similarity_matrix[local_index, local_index]
                    ),
                }
            )

    pointwise = pd.DataFrame(pointwise_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    samplewise = pd.DataFrame(samplewise_rows)
    pairwise = pd.DataFrame(pairwise_rows)

    return pointwise, retrieval, samplewise, pairwise


# =============================================================================
# Publication-style summaries
# =============================================================================

def add_average_row_pointwise(df: pd.DataFrame) -> pd.DataFrame:
    average = {
        "channel": "Average",
        "channel_name": "Average",
        "n_valid": "",
        "cosine_mean": float(df["cosine_mean"].mean()),
        "cosine_median": float(df["cosine_median"].mean()),
        "cosine_p25": float(df["cosine_p25"].mean()),
    }
    return pd.concat(
        [df, pd.DataFrame([average])],
        ignore_index=True,
    )


def add_average_row_retrieval(df: pd.DataFrame) -> pd.DataFrame:
    average = {
        "channel": "Average",
        "channel_name": "Average",
        "n_valid": "",
        "MRR": float(df["MRR"].mean()),
        "median_rank": float(df["median_rank"].mean()),
        "Recall@3": float(df["Recall@3"].mean()),
        "Recall@5": float(df["Recall@5"].mean()),
    }
    return pd.concat(
        [df, pd.DataFrame([average])],
        ignore_index=True,
    )


# =============================================================================
# Figures
# =============================================================================

def plot_fig3_cosine_distributions(
    samplewise: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Reproduce the channel-wise CS distributions underlying Fig. 3.
    """
    data = [
        samplewise.loc[
            samplewise["channel"].eq(channel),
            "CS",
        ].to_numpy(dtype=float)
        for channel in SELECTED_CHANNELS
    ]

    fig, ax = plt.subplots(figsize=(10.0, 5.5))

    ax.boxplot(
        data,
        labels=[
            f"Ch{channel}"
            for channel in SELECTED_CHANNELS
        ],
        showfliers=True,
    )

    ax.set_xlabel("Mechanism channel")
    ax.set_ylabel("Cosine similarity (CS)")
    ax.set_ylim(-1.05, 1.05)

    fig.tight_layout()
    fig.savefig(
        output_dir / "fig3_cosine_distributions.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "fig3_cosine_distributions.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_fig4_pairwise_similarity(
    pairwise: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Reproduce the six channel-wise pairwise-similarity panels underlying Fig. 4.
    """
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.0, 7.5),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    for axis, channel in zip(axes, SELECTED_CHANNELS):
        subset = pairwise.loc[
            pairwise["channel"].eq(channel)
        ]

        x = subset["reference_similarity"].to_numpy(dtype=float)
        y = subset["predicted_similarity"].to_numpy(dtype=float)

        axis.scatter(
            x,
            y,
            s=10,
            alpha=0.5,
            edgecolors="none",
        )
        axis.plot(
            [-1, 1],
            [-1, 1],
            linestyle="--",
            linewidth=1.0,
        )

        rho = float(
            subset["spearman_rho_channel"].iloc[0]
        )

        axis.text(
            0.04,
            0.94,
            f"$\\rho$ = {rho:.3f}",
            transform=axis.transAxes,
            va="top",
        )

        axis.set_title(
            f"Ch{channel}: {SHORT_CHANNEL_NAMES[channel]}"
        )
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)

    for axis in axes[3:]:
        axis.set_xlabel("Reference pairwise CS")

    axes[0].set_ylabel("Predicted pairwise CS")
    axes[3].set_ylabel("Predicted pairwise CS")

    fig.tight_layout()
    fig.savefig(
        output_dir / "fig4_pairwise_similarity.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "fig4_pairwise_similarity.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final mechanism-pretraining checkpoint "
            "on the held-out test set."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help=(
            "Pretraining run directory containing the checkpoint, "
            "saved scaler, resolved config, and split identity."
        ),
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
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        type=Path,
        help=(
            "Root directory produced by build_mechanism_embeddings.py."
        ),
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

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    df, x_scaled, targets, masks = prepare_test_data(
        structured_data=args.structured_data,
        sheet_name=args.sheet_name,
        reference_root=args.reference_root,
        split_csv=run_dir / "data_split.csv",
        scaler_path=run_dir / "feature_scaler.joblib",
    )

    model = load_model(
        run_dir=run_dir,
        device=device,
    )

    predictions = predict_all(
        model=model,
        x_scaled=x_scaled,
        batch_size=args.batch_size,
        device=device,
    )

    pointwise, retrieval, samplewise, pairwise = evaluate_channels(
        df=df,
        predictions=predictions,
        targets=targets,
        masks=masks,
    )

    # ------------------------------------------------------------------
    # Table 1 and Table 2 source data
    # ------------------------------------------------------------------
    table1 = add_average_row_pointwise(pointwise)
    table2 = add_average_row_retrieval(retrieval)

    table1.to_csv(
        output_dir / "table1_pointwise_cs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table2.to_csv(
        output_dir / "table2_reference_retrieval.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Figure and sample-level source data
    # ------------------------------------------------------------------
    samplewise.to_csv(
        output_dir / "test_samplewise_mechanism_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairwise.to_csv(
        output_dir / "fig4_pairwise_similarity_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pointwise.to_csv(
        output_dir / "fig3_channel_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_fig3_cosine_distributions(
        samplewise=samplewise,
        output_dir=output_dir,
    )
    plot_fig4_pairwise_similarity(
        pairwise=pairwise,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # Compact machine-readable summary
    # ------------------------------------------------------------------
    summary = {
        "evaluation_split": "test",
        "test_rows_total": int(df["split"].eq("test").sum()),
        "channel_averaged_mean_CS": float(
            pointwise["cosine_mean"].mean()
        ),
        "channel_averaged_median_CS": float(
            pointwise["cosine_median"].mean()
        ),
        "channel_averaged_MRR": float(
            retrieval["MRR"].mean()
        ),
        "channel_averaged_median_rank": float(
            retrieval["median_rank"].mean()
        ),
        "channel_averaged_Recall@3": float(
            retrieval["Recall@3"].mean()
        ),
        "channel_averaged_Recall@5": float(
            retrieval["Recall@5"].mean()
        ),
        "channels": {
            str(channel): {
                "name": CHANNEL_NAMES[channel],
                "n_valid": int(
                    pointwise.loc[
                        pointwise["channel"].eq(channel),
                        "n_valid",
                    ].iloc[0]
                ),
                "mean_CS": float(
                    pointwise.loc[
                        pointwise["channel"].eq(channel),
                        "cosine_mean",
                    ].iloc[0]
                ),
                "median_CS": float(
                    pointwise.loc[
                        pointwise["channel"].eq(channel),
                        "cosine_median",
                    ].iloc[0]
                ),
                "P25_CS": float(
                    pointwise.loc[
                        pointwise["channel"].eq(channel),
                        "cosine_p25",
                    ].iloc[0]
                ),
                "pairwise_spearman_rho": float(
                    pairwise.loc[
                        pairwise["channel"].eq(channel),
                        "spearman_rho_channel",
                    ].iloc[0]
                ),
                "MRR": float(
                    retrieval.loc[
                        retrieval["channel"].eq(channel),
                        "MRR",
                    ].iloc[0]
                ),
                "median_rank": float(
                    retrieval.loc[
                        retrieval["channel"].eq(channel),
                        "median_rank",
                    ].iloc[0]
                ),
                "Recall@3": float(
                    retrieval.loc[
                        retrieval["channel"].eq(channel),
                        "Recall@3",
                    ].iloc[0]
                ),
                "Recall@5": float(
                    retrieval.loc[
                        retrieval["channel"].eq(channel),
                        "Recall@5",
                    ].iloc[0]
                ),
            }
            for channel in SELECTED_CHANNELS
        },
    }

    with (
        output_dir
        / "test_evaluation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 88)
    print("Final mechanism-pretraining test evaluation")
    print("=" * 88)

    for channel in SELECTED_CHANNELS:
        row1 = pointwise.loc[
            pointwise["channel"].eq(channel)
        ].iloc[0]
        row2 = retrieval.loc[
            retrieval["channel"].eq(channel)
        ].iloc[0]
        rho = pairwise.loc[
            pairwise["channel"].eq(channel),
            "spearman_rho_channel",
        ].iloc[0]

        print(
            f"Ch{channel}: "
            f"mean CS={row1['cosine_mean']:.3f}, "
            f"median={row1['cosine_median']:.3f}, "
            f"rho={rho:.3f}, "
            f"MRR={row2['MRR']:.3f}, "
            f"R@5={row2['Recall@5']:.3f}"
        )

    print("-" * 88)
    print(
        f"Channel-averaged mean CS:      "
        f"{summary['channel_averaged_mean_CS']:.3f}"
    )
    print(
        f"Channel-averaged median CS:    "
        f"{summary['channel_averaged_median_CS']:.3f}"
    )
    print(
        f"Channel-averaged MRR:          "
        f"{summary['channel_averaged_MRR']:.3f}"
    )
    print(
        f"Channel-averaged median rank:  "
        f"{summary['channel_averaged_median_rank']:.3f}"
    )
    print(
        f"Channel-averaged Recall@5:     "
        f"{summary['channel_averaged_Recall@5']:.3f}"
    )
    print(f"Outputs: {output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
