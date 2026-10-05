#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prospective mechanism-aware screening of kinetically similar Co-based alloy states.

Paper-release implementation
----------------------------
This script reproduces the screening workflow described in Section 3.5 and the
source data underlying Fig. 7, Fig. S10, Table 5, and Table S5.

The manuscript workflow is fixed as follows:

1. Candidate generation
   - Generate 4,000 Latin-hypercube samples over the eight searched alloying
     variables.
   - Discretize searched compositions to 0.5 at.% increments.
   - Recalculate Co as the balance element.
   - Remove infeasible and duplicate compositions.
   - The fixed search specification yields 1,667 feasible candidate states.

2. Frozen-transfer inference
   - Use the final frozen-transfer model to predict lg(kp) and six 64-D
     mechanism representations for every candidate.

3. Kinetic pre-screen
   - Consider all unordered candidate pairs.
   - Retain pairs satisfying |Delta lg(kp)| <= 0.10.

4. Channel-normalized mechanism separation
   - For each mechanism channel c:
         d_c = 1 - CS_c
   - Convert d_c to its empirical percentile q_c within the CROSS-SOURCE
     pairwise cosine-distance distribution of the TRAINING-reference mechanism
     library for that channel.

5. Balanced multichannel divergence
       N_eff = (sum_c q_c)^2 / sum_c q_c^2
       D_mech = mean(q_c) * sqrt(N_eff / 6)

6. Target region
   - Tight kinetic criterion: |Delta lg(kp)| <= 0.05
   - Mechanism criterion:      D_mech >= 0.60

IMPORTANT
---------
No additional hidden-state selection rules are used for the manuscript target
region. In particular, channel-count, mean-percentile, dominant-channel, or
N_eff thresholds are NOT used to determine whether a pair belongs to the final
target region. q_c >= 0.75 is retained only as a channel-level visualization
indicator for Fig. 7(d).

The representative Candidate A / Candidate B pair used in the manuscript is
specified by composition and is verified against the final target-region table.
It is not selected by an additional optimization score.

Inputs
------
<pretrain_run>/
    feature_scaler.joblib
    resolved_pretraining_config.json
    best_model_state.pt

<transfer_run>/
    best_transfer_model_state.pt
    resolved_transfer_config.json

<reference_root>/channel_1/reference_embeddings_64d.csv
...
<reference_root>/channel_6/reference_embeddings_64d.csv

Outputs
-------
all_screened_candidates.csv
all_candidate_predicted_mechanism_vectors.csv
mechanism_distance_calibration_summary.csv
fig7a_funnel_counts.csv
fig7b_pair_landscape.csv
kinetically_matched_pair_diagnostics.csv
final_target_region_pairs.csv
fig7c_representative_pair_composition.csv
fig7d_representative_pair_channel_divergence.csv
representative_pair.json
tableS5_top5_training_reference_retrieval.csv
figS10_umap_training_reference_coordinates.csv
figS10_candidate_overlays.csv
figS10_representative_pair_umap_preview.png/.pdf
screening_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "screen_mechanism_divergence.py requires umap-learn. "
        "Install it with: pip install umap-learn"
    ) from exc

from pretrain_mechanism_encoder import (
    CHANNEL_NAMES,
    EMBED_DIM,
    FEATURE_COLS,
    SELECTED_CHANNELS,
    resolve_device,
)
from evaluate_kinetics import load_transfer_model


# =============================================================================
# Fixed manuscript screening specification
# =============================================================================

CHANNELS: Tuple[int, ...] = tuple(int(x) for x in SELECTED_CHANNELS)

CHANNEL_SHORT_NAMES: Dict[int, str] = {
    1: "Protective scale formation",
    2: "Internal/selective oxidation",
    3: "Non-protective/transient oxide formation",
    4: "Transport/defect kinetics",
    5: "Scale failure/spallation",
    6: "Alloying-element interaction",
}

# Initial Latin-hypercube population.
N_LHS_SAMPLES = 4000
RANDOM_STATE = 42
COMPOSITION_STEP = 0.5

# Manuscript search ranges, at.%.
SEARCH_RANGES: Dict[str, Tuple[float, float]] = {
    "Al": (7.0, 15.0),
    "Cr": (3.0, 15.0),
    "Ni": (20.0, 35.0),
    "W": (0.0, 8.0),
    "Ti": (0.0, 8.0),
    "Mo": (0.0, 4.0),
    "Ta": (0.0, 6.0),
    "Si": (0.0, 3.0),
}

COMPOSITION_COLS: Tuple[str, ...] = (
    "Co", "Al", "W", "Ni", "Cr", "Mo", "Fe", "Nb", "C", "Hf",
    "Si", "Ta", "Ti", "Y", "V", "B", "Zr", "Mn", "Sc", "La", "Re",
)

BALANCE_ELEMENT = "Co"
TOTAL_COMPOSITION = 100.0

# Feasibility range for the Co balance element. For the fixed random seed and
# search ranges, Co never exceeds 61 at.% after the lower-bound filter; the
# upper limit therefore does not affect the reported 1,667-state population.
MIN_CO = 40.0
MAX_CO = 80.0
EXPECTED_FEASIBLE_CANDIDATES = 1667

# Fixed processing / oxidation condition used in the prospective screen.
BASE_FEATURES: Dict[str, float] = {
    "solu_temp": 1200.0,
    "solu_time": 12.0,
    "middle_temp": 0.0,
    "middle_time": 0.0,
    "aging_temp": 900.0,
    "aging_time": 100.0,
    "Co": 0.0,  # recalculated below
    "Al": 0.0,
    "W": 0.0,
    "Ni": 0.0,
    "Cr": 0.0,
    "Mo": 0.0,
    "Fe": 0.0,
    "Nb": 0.0,
    "C": 0.0,
    "Hf": 0.0,
    "Si": 0.0,
    "Ta": 0.0,
    "Ti": 0.0,
    "Y": 0.0,
    "V": 0.0,
    "B": 0.0,
    "Zr": 0.0,
    "Mn": 0.0,
    "Sc": 0.0,
    "La": 0.0,
    "Re": 0.0,
    "Temperature": 900.0,
    "E_Time": 100.0,
}

# Screening thresholds reported in the manuscript.
KINETIC_PRESCREEN_TOLERANCE = 0.10
TIGHT_KINETIC_TOLERANCE = 0.05
D_MECH_THRESHOLD = 0.60

# Used only to highlight strongly separated individual channels in Fig. 7(d).
# This is NOT a pair-selection criterion.
HIGH_CHANNEL_PERCENTILE = 0.75

# Reference-distance calibration uses cross-publication TRAIN-reference pairs.
CROSS_SOURCE_CALIBRATION = True

# Fixed UMAP settings for representative-candidate visualization (Fig. S10).
UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.15
UMAP_METRIC = "cosine"

# Candidate A and Candidate B reported in Section 3.5.
REPRESENTATIVE_A: Dict[str, float] = {
    "Co": 44.5,
    "Ni": 23.0,
    "Cr": 12.5,
    "Al": 10.5,
    "W": 8.0,
    "Ti": 0.5,
    "Mo": 0.5,
    "Ta": 0.0,
    "Si": 0.5,
}

REPRESENTATIVE_B: Dict[str, float] = {
    "Co": 50.0,
    "Ni": 25.0,
    "Cr": 5.0,
    "Al": 11.5,
    "W": 1.5,
    "Ti": 0.0,
    "Mo": 2.5,
    "Ta": 4.0,
    "Si": 0.5,
}


# =============================================================================
# Generic helpers
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


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Zero-norm mechanism representation encountered.")

    return x / np.clip(norms, eps, None)


def validate_feature_definition() -> None:
    missing = [column for column in FEATURE_COLS if column not in BASE_FEATURES]
    if missing:
        raise ValueError(f"BASE_FEATURES is missing model inputs: {missing}")

    unknown = sorted(set(SEARCH_RANGES) - set(FEATURE_COLS))
    if unknown:
        raise ValueError(f"SEARCH_RANGES contains unknown model inputs: {unknown}")

    if BALANCE_ELEMENT in SEARCH_RANGES:
        raise ValueError("Co must be recalculated as the balance element, not sampled.")


# =============================================================================
# 1. Candidate-space construction
# =============================================================================


def generate_candidate_space() -> pd.DataFrame:
    validate_feature_definition()

    variable_columns = list(SEARCH_RANGES)

    sampler = qmc.LatinHypercube(
        d=len(variable_columns),
        seed=RANDOM_STATE,
    )
    unit_samples = sampler.random(n=N_LHS_SAMPLES)

    lower = np.asarray(
        [SEARCH_RANGES[column][0] for column in variable_columns],
        dtype=float,
    )
    upper = np.asarray(
        [SEARCH_RANGES[column][1] for column in variable_columns],
        dtype=float,
    )

    sampled = qmc.scale(unit_samples, lower, upper)
    df = pd.DataFrame(sampled, columns=variable_columns)

    # Fixed 0.5 at.% discretization.
    for column in variable_columns:
        df[column] = (
            np.round(df[column] / COMPOSITION_STEP)
            * COMPOSITION_STEP
        )

    # Add every non-searched structured model input at its fixed value.
    for column in FEATURE_COLS:
        if column not in df.columns:
            df[column] = float(BASE_FEATURES[column])

    # Recalculate Co as the balance element.
    non_co = [column for column in COMPOSITION_COLS if column != "Co"]
    df["Co"] = TOTAL_COMPOSITION - df[non_co].sum(axis=1)

    feasible = (
        df["Co"].between(MIN_CO, MAX_CO)
        & np.isfinite(df[list(FEATURE_COLS)]).all(axis=1)
    )

    df = (
        df.loc[feasible]
        .copy()
        .drop_duplicates(subset=list(FEATURE_COLS))
        .reset_index(drop=True)
    )

    if len(df) != EXPECTED_FEASIBLE_CANDIDATES:
        raise RuntimeError(
            "The fixed screening specification no longer reproduces the "
            f"manuscript candidate count: expected {EXPECTED_FEASIBLE_CANDIDATES}, "
            f"obtained {len(df)}. Check the search ranges, rounding rule, Co "
            "feasibility range, and random seed before continuing."
        )

    df.insert(
        0,
        "candidate_id",
        [f"SYN{index:04d}" for index in range(len(df))],
    )

    # Reorder for human-readable outputs while preserving all model features.
    front = [
        "candidate_id",
        "Co", "Ni", "Cr", "Al", "W", "Ti", "Mo", "Ta", "Si",
    ]
    remaining = [
        column for column in df.columns
        if column not in front
    ]
    return df[front + remaining]


# =============================================================================
# 2. Frozen-transfer candidate inference
# =============================================================================


@torch.no_grad()
def predict_candidates(
    candidates: pd.DataFrame,
    model: torch.nn.Module,
    scaler: Any,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    x_raw = candidates[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    prediction_parts: List[np.ndarray] = []
    mechanism_parts: Dict[int, List[np.ndarray]] = {
        channel: [] for channel in CHANNELS
    }

    model.eval()

    for start in range(0, len(x_scaled), batch_size):
        xb = torch.as_tensor(
            x_scaled[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )

        outputs = model(xb)
        prediction_parts.append(
            outputs["lgkp"].cpu().numpy()
        )

        for channel in CHANNELS:
            mechanism_parts[channel].append(
                outputs["mechanism_vectors"][str(channel)]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    lgkp = np.concatenate(prediction_parts).astype(np.float64)
    mechanism_vectors = {
        channel: np.vstack(parts).astype(np.float32)
        for channel, parts in mechanism_parts.items()
    }

    return lgkp, mechanism_vectors


def export_candidate_vectors(
    candidates: pd.DataFrame,
    mechanism_vectors: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    identity = candidates[["candidate_id"]].copy()

    for channel in CHANNELS:
        part = identity.copy()
        part["channel"] = channel
        part["channel_name"] = CHANNEL_NAMES[channel]

        vector_df = pd.DataFrame(
            mechanism_vectors[channel],
            columns=[f"emb_{dim}" for dim in range(EMBED_DIM)],
        )

        rows.append(
            pd.concat([part, vector_df], axis=1)
        )

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# 3. TRAIN-reference mechanism spaces and distance calibration
# =============================================================================


def load_reference_space(
    reference_root: Path,
    channel: int,
) -> Dict[str, Any]:
    path = (
        reference_root
        / f"channel_{channel}"
        / "reference_embeddings_64d.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Channel-{channel} reference file not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "Ref_source_num",
        "sample_id",
        "alloy_id",
        "condition_id",
        "split",
        "embedding_valid",
        "canonical_tags",
        "mechanism_text",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Channel-{channel} reference file is missing columns: {missing}"
        )

    vector_columns = [f"emb_{dim}" for dim in range(EMBED_DIM)]
    missing_vectors = [
        column for column in vector_columns
        if column not in df.columns
    ]
    if missing_vectors:
        raise ValueError(
            f"Channel-{channel} reference file is missing 64-D vectors."
        )

    df["split"] = (
        df["split"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["embedding_valid"] = pd.to_numeric(
        df["embedding_valid"],
        errors="raise",
    ).astype(int)

    training = df.loc[
        df["embedding_valid"].eq(1)
        & df["split"].eq("train")
    ].copy().reset_index(drop=True)

    if len(training) < 3:
        raise ValueError(
            f"Channel-{channel} has too few valid training-reference states."
        )

    matrix = training[vector_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Channel-{channel} training-reference vectors contain NaN/inf."
        )

    return {
        "path": path,
        "training": training,
        "vector_columns": vector_columns,
        "matrix": matrix,
        "matrix_normalized": l2_normalize(matrix),
    }


def pairwise_cross_source_distances(
    normalized_matrix: np.ndarray,
    source_ids: Sequence[str],
) -> np.ndarray:
    n = len(normalized_matrix)
    pair_i, pair_j = np.triu_indices(n, k=1)

    cosine = np.sum(
        normalized_matrix[pair_i]
        * normalized_matrix[pair_j],
        axis=1,
    )
    distance = 1.0 - np.clip(cosine, -1.0, 1.0)

    if CROSS_SOURCE_CALIBRATION:
        sources = np.asarray(
            [str(value) for value in source_ids],
            dtype=object,
        )
        keep = sources[pair_i] != sources[pair_j]
        distance = distance[keep]

    distance = distance[np.isfinite(distance)]
    if len(distance) == 0:
        raise ValueError("Reference calibration produced no valid distances.")

    return np.sort(distance.astype(np.float64))


def build_reference_calibration(
    reference_root: Path,
) -> Tuple[
    Dict[int, Dict[str, Any]],
    Dict[int, np.ndarray],
    pd.DataFrame,
]:
    spaces: Dict[int, Dict[str, Any]] = {}
    calibration: Dict[int, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    for channel in CHANNELS:
        space = load_reference_space(
            reference_root=reference_root,
            channel=channel,
        )
        spaces[channel] = space

        training = space["training"]
        distances = pairwise_cross_source_distances(
            normalized_matrix=space["matrix_normalized"],
            source_ids=training["Ref_source_num"].astype(str).tolist(),
        )
        calibration[channel] = distances

        rows.append(
            {
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "n_training_reference_states": int(len(training)),
                "n_cross_source_reference_pairs": int(len(distances)),
                "distance_q25": float(np.quantile(distances, 0.25)),
                "distance_q50": float(np.quantile(distances, 0.50)),
                "distance_q75": float(np.quantile(distances, 0.75)),
                "distance_q90": float(np.quantile(distances, 0.90)),
                "distance_q95": float(np.quantile(distances, 0.95)),
                "distance_max": float(np.max(distances)),
            }
        )

    return spaces, calibration, pd.DataFrame(rows)


# =============================================================================
# 4. Pair construction and multichannel divergence
# =============================================================================


def build_all_unordered_pairs(n_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
    pair_i, pair_j = np.triu_indices(n_candidates, k=1)
    return pair_i.astype(np.int32), pair_j.astype(np.int32)


def score_mechanism_divergence(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    mechanism_vectors: Mapping[int, np.ndarray],
    calibration: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    n_pairs = len(pair_i)
    q = np.zeros((n_pairs, len(CHANNELS)), dtype=np.float32)
    distance = np.zeros((n_pairs, len(CHANNELS)), dtype=np.float32)

    for column, channel in enumerate(CHANNELS):
        vectors = l2_normalize(mechanism_vectors[channel])

        cosine = np.sum(
            vectors[pair_i] * vectors[pair_j],
            axis=1,
        )
        cosine = np.clip(cosine, -1.0, 1.0)
        channel_distance = 1.0 - cosine
        distance[:, column] = channel_distance.astype(np.float32)

        reference_distribution = calibration[channel]
        percentile = (
            np.searchsorted(
                reference_distribution,
                channel_distance,
                side="right",
            )
            / len(reference_distribution)
        )
        q[:, column] = percentile.astype(np.float32)

    mean_q = q.mean(axis=1)
    sum_q = q.sum(axis=1)
    sum_q_squared = np.sum(q ** 2, axis=1)

    n_eff = np.divide(
        sum_q ** 2,
        sum_q_squared,
        out=np.zeros_like(sum_q),
        where=sum_q_squared > 1e-12,
    )

    balance_factor = np.sqrt(
        np.clip(n_eff / len(CHANNELS), 0.0, 1.0)
    )
    d_mech = mean_q * balance_factor

    output = pd.DataFrame(
        {
            "q_mean": mean_q,
            "N_eff": n_eff,
            "D_mech": d_mech,
        }
    )

    for column, channel in enumerate(CHANNELS):
        output[f"ch{channel}_cosine_distance"] = distance[:, column]
        output[f"ch{channel}_q"] = q[:, column]

    return output


def build_screened_pair_table(
    candidates: pd.DataFrame,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    delta_lgkp: np.ndarray,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    a = candidates.iloc[pair_i].reset_index(drop=True)
    b = candidates.iloc[pair_j].reset_index(drop=True)

    output = pd.DataFrame(
        {
            "candidate_a": a["candidate_id"].to_numpy(),
            "candidate_b": b["candidate_id"].to_numpy(),
            "lgkp_a": a["lgkp_pred"].to_numpy(dtype=float),
            "lgkp_b": b["lgkp_pred"].to_numpy(dtype=float),
            "abs_delta_lgkp": delta_lgkp.astype(float),
        }
    )

    output = pd.concat(
        [output, scores.reset_index(drop=True)],
        axis=1,
    )

    output["passes_tight_kinetic_criterion"] = (
        output["abs_delta_lgkp"] <= TIGHT_KINETIC_TOLERANCE
    )
    output["passes_D_mech_threshold"] = (
        output["D_mech"] >= D_MECH_THRESHOLD
    )

    # This conjunction is the manuscript target-region definition.
    output["in_final_target_region"] = (
        output["passes_tight_kinetic_criterion"]
        & output["passes_D_mech_threshold"]
    )

    return output


# =============================================================================
# 5. Representative Candidate A / Candidate B
# =============================================================================


def find_candidate_by_composition(
    candidates: pd.DataFrame,
    target: Mapping[str, float],
    label: str,
) -> pd.Series:
    mask = np.ones(len(candidates), dtype=bool)

    for column, expected in target.items():
        if column not in candidates.columns:
            raise ValueError(
                f"Representative composition uses unknown column {column!r}."
            )
        values = pd.to_numeric(
            candidates[column],
            errors="coerce",
        ).to_numpy(dtype=float)
        mask &= np.isclose(values, float(expected), atol=1e-9)

    matched = candidates.loc[mask]

    if len(matched) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} composition in the fixed candidate "
            f"space; found {len(matched)}."
        )

    return matched.iloc[0]


def find_representative_pair_row(
    pairs: pd.DataFrame,
    candidate_a: str,
    candidate_b: str,
) -> pd.Series:
    mask = (
        pairs["candidate_a"].eq(candidate_a)
        & pairs["candidate_b"].eq(candidate_b)
    ) | (
        pairs["candidate_a"].eq(candidate_b)
        & pairs["candidate_b"].eq(candidate_a)
    )

    matched = pairs.loc[mask]

    if len(matched) != 1:
        raise RuntimeError(
            "The manuscript representative pair was not found uniquely within "
            "the |Delta lg(kp)| <= 0.10 pair table."
        )

    row = matched.iloc[0]

    if not bool(row["in_final_target_region"]):
        raise RuntimeError(
            "The manuscript representative pair no longer lies in the final "
            "target region under the current model outputs. Update the model/results "
            "or manuscript instead of forcing the pair into the target set."
        )

    return row


def build_representative_composition_table(
    candidate_a: pd.Series,
    candidate_b: pd.Series,
) -> pd.DataFrame:
    columns = [
        "Co", "Ni", "Cr", "Al", "W", "Ti", "Mo", "Ta", "Si",
    ]

    rows: List[Dict[str, Any]] = []
    for label, candidate in (
        ("Candidate A", candidate_a),
        ("Candidate B", candidate_b),
    ):
        row: Dict[str, Any] = {
            "candidate": label,
            "candidate_id": candidate["candidate_id"],
            "lgkp_pred": float(candidate["lgkp_pred"]),
        }
        for column in columns:
            row[column] = float(candidate[column])
        rows.append(row)

    return pd.DataFrame(rows)


def build_representative_channel_table(
    representative_pair: pd.Series,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for channel in CHANNELS:
        q_value = float(representative_pair[f"ch{channel}_q"])
        distance = float(
            representative_pair[f"ch{channel}_cosine_distance"]
        )

        rows.append(
            {
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "cosine_distance": distance,
                "cosine_similarity": 1.0 - distance,
                "q_c": q_value,
                "q_c_ge_0.75": bool(
                    q_value >= HIGH_CHANNEL_PERCENTILE
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 6. Representative-pair Top-5 retrieval and Fig. S10 UMAP data
# =============================================================================


def retrieve_top5_training_references(
    query_vector: np.ndarray,
    space: Mapping[str, Any],
) -> pd.DataFrame:
    query = l2_normalize(query_vector)[0]
    similarity = space["matrix_normalized"] @ query
    order = np.argsort(-similarity)[:5]

    top = space["training"].iloc[order].copy().reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, 6, dtype=int))
    top.insert(1, "CS_P_R", similarity[order].astype(float))
    top["library_index"] = order.astype(int)
    return top


def fit_representative_umaps(
    spaces: Mapping[int, Mapping[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    for channel in CHANNELS:
        space = spaces[channel]
        n_neighbors = min(
            UMAP_N_NEIGHBORS,
            len(space["training"]) - 1,
        )

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=RANDOM_STATE,
            transform_seed=RANDOM_STATE,
        )

        coordinates = reducer.fit_transform(
            space["matrix"]
        ).astype(np.float32)

        results[channel] = {
            "reducer": reducer,
            "coordinates": coordinates,
        }

    return results


def build_representative_retrieval_and_umap(
    candidate_a: pd.Series,
    candidate_b: pd.Series,
    mechanism_vectors: Mapping[int, np.ndarray],
    spaces: Mapping[int, Mapping[str, Any]],
    umaps: Mapping[int, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_index = {
        str(candidate_a["candidate_id"]): int(candidate_a.name),
        str(candidate_b["candidate_id"]): int(candidate_b.name),
    }

    candidate_labels = {
        str(candidate_a["candidate_id"]): "Candidate A",
        str(candidate_b["candidate_id"]): "Candidate B",
    }

    retrieval_rows: List[Dict[str, Any]] = []
    overlay_rows: List[Dict[str, Any]] = []
    reference_frames: List[pd.DataFrame] = []

    for channel in CHANNELS:
        space = spaces[channel]
        reducer = umaps[channel]["reducer"]
        coordinates = umaps[channel]["coordinates"]

        training = space["training"].copy().reset_index(drop=True)
        frame = training[
            [
                "Ref_source_num",
                "sample_id",
                "alloy_id",
                "condition_id",
                "canonical_tags",
                "mechanism_text",
            ]
        ].copy()
        frame.insert(0, "channel", channel)
        frame.insert(1, "channel_name", CHANNEL_NAMES[channel])
        frame["umap_x"] = coordinates[:, 0]
        frame["umap_y"] = coordinates[:, 1]
        reference_frames.append(frame)

        for candidate_id, row_index in candidate_index.items():
            label = candidate_labels[candidate_id]
            query = mechanism_vectors[channel][row_index]

            top = retrieve_top5_training_references(
                query_vector=query,
                space=space,
            )

            query_xy = reducer.transform(
                query.reshape(1, -1)
            )[0]

            overlay_rows.append(
                {
                    "candidate": label,
                    "candidate_id": candidate_id,
                    "channel": channel,
                    "channel_name": CHANNEL_NAMES[channel],
                    "point_type": "prediction",
                    "rank": np.nan,
                    "CS_P_R": np.nan,
                    "umap_x": float(query_xy[0]),
                    "umap_y": float(query_xy[1]),
                }
            )

            for _, neighbor in top.iterrows():
                lib_index = int(neighbor["library_index"])
                xy = coordinates[lib_index]

                retrieval_rows.append(
                    {
                        "candidate": label,
                        "candidate_id": candidate_id,
                        "channel": channel,
                        "channel_name": CHANNEL_NAMES[channel],
                        "rank": int(neighbor["rank"]),
                        "CS_P_R": float(neighbor["CS_P_R"]),
                        "Ref_source_num": str(neighbor["Ref_source_num"]),
                        "sample_id": str(neighbor["sample_id"]),
                        "alloy_id": str(neighbor["alloy_id"]),
                        "condition_id": str(neighbor["condition_id"]),
                        "canonical_tags": str(neighbor["canonical_tags"]),
                        "mechanism_text": str(neighbor["mechanism_text"]),
                    }
                )

                overlay_rows.append(
                    {
                        "candidate": label,
                        "candidate_id": candidate_id,
                        "channel": channel,
                        "channel_name": CHANNEL_NAMES[channel],
                        "point_type": "top5_training_reference",
                        "rank": int(neighbor["rank"]),
                        "CS_P_R": float(neighbor["CS_P_R"]),
                        "umap_x": float(xy[0]),
                        "umap_y": float(xy[1]),
                    }
                )

    return (
        pd.DataFrame(retrieval_rows),
        pd.DataFrame(overlay_rows),
        pd.concat(reference_frames, ignore_index=True),
    )


def plot_representative_umap_preview(
    reference_coordinates: pd.DataFrame,
    overlays: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        6,
        figsize=(18.0, 6.2),
        squeeze=False,
    )

    candidate_labels = ("Candidate A", "Candidate B")

    for row_index, candidate_label in enumerate(candidate_labels):
        for column_index, channel in enumerate(CHANNELS):
            axis = axes[row_index][column_index]

            background = reference_coordinates.loc[
                reference_coordinates["channel"].eq(channel)
            ]
            overlay = overlays.loc[
                overlays["channel"].eq(channel)
                & overlays["candidate"].eq(candidate_label)
            ]

            axis.scatter(
                background["umap_x"],
                background["umap_y"],
                s=9,
                alpha=0.22,
                linewidths=0,
            )

            neighbours = overlay.loc[
                overlay["point_type"].eq("top5_training_reference")
            ]
            axis.scatter(
                neighbours["umap_x"],
                neighbours["umap_y"],
                s=42,
                marker="o",
            )

            for _, point in neighbours.iterrows():
                axis.text(
                    float(point["umap_x"]),
                    float(point["umap_y"]),
                    str(int(point["rank"])),
                    fontsize=7,
                    ha="center",
                    va="center",
                )

            prediction = overlay.loc[
                overlay["point_type"].eq("prediction")
            ]
            axis.scatter(
                prediction["umap_x"],
                prediction["umap_y"],
                s=100,
                marker="*",
            )

            if row_index == 0:
                axis.set_title(
                    f"Ch{channel}\n{CHANNEL_SHORT_NAMES[channel]}",
                    fontsize=9,
                )

            if column_index == 0:
                axis.set_ylabel(candidate_label)

            axis.set_xticks([])
            axis.set_yticks([])

    fig.tight_layout()
    fig.savefig(
        output_dir / "figS10_representative_pair_umap_preview.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "figS10_representative_pair_umap_preview.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 7. Compact Fig. 7 preview/source-data outputs
# =============================================================================


def build_funnel_counts(
    n_candidates: int,
    n_all_pairs: int,
    n_prescreen: int,
    n_tight: int,
    n_target: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "Candidate states",
                "criterion": "Feasible unique candidate compositions",
                "count": n_candidates,
            },
            {
                "stage": "All unordered pairs",
                "criterion": "C(n,2)",
                "count": n_all_pairs,
            },
            {
                "stage": "Kinetic pre-screen",
                "criterion": "|Delta lgkp| <= 0.10",
                "count": n_prescreen,
            },
            {
                "stage": "Tight kinetic subset",
                "criterion": "|Delta lgkp| <= 0.05",
                "count": n_tight,
            },
            {
                "stage": "Final target region",
                "criterion": "|Delta lgkp| <= 0.05 and D_mech >= 0.60",
                "count": n_target,
            },
        ]
    )


def make_pair_landscape_preview(
    pairs: pd.DataFrame,
    representative_pair: pd.Series,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 5.0))

    ax.scatter(
        pairs["abs_delta_lgkp"],
        pairs["D_mech"],
        s=10,
        alpha=0.35,
        linewidths=0,
    )

    ax.scatter(
        [float(representative_pair["abs_delta_lgkp"])],
        [float(representative_pair["D_mech"])],
        s=100,
        marker="*",
    )

    ax.axvline(TIGHT_KINETIC_TOLERANCE, linestyle="--", linewidth=1.0)
    ax.axhline(D_MECH_THRESHOLD, linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"$|\Delta\,\mathrm{lg}(k_p)|$")
    ax.set_ylabel(r"$D_{\mathrm{mech}}$")

    fig.tight_layout()
    fig.savefig(
        output_dir / "fig7b_pair_landscape_preview.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "fig7b_pair_landscape_preview.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen kinetically similar candidate alloys for "
            "multichannel mechanism divergence."
        )
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
        "--reference-root",
        required=True,
        type=Path,
        help="Root produced by build_mechanism_embeddings.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mechanism_screening"),
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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_run = args.pretrain_run.resolve()
    transfer_run = args.transfer_run.resolve()
    reference_root = args.reference_root.resolve()

    device = resolve_device(args.device)

    print("=" * 100)
    print("Prospective mechanism-aware screening")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Fixed prospective candidate space
    # ------------------------------------------------------------------
    candidates = generate_candidate_space()
    n_candidates = len(candidates)

    pair_i_all, pair_j_all = build_all_unordered_pairs(n_candidates)
    n_all_pairs = len(pair_i_all)

    print(f"Feasible candidate states: {n_candidates:,}")
    print(f"All unordered pairs:       {n_all_pairs:,}")

    # ------------------------------------------------------------------
    # 2. Frozen-transfer inference
    # ------------------------------------------------------------------
    scaler_path = pretrain_run / "feature_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Pretraining feature scaler not found: {scaler_path}"
        )
    scaler = joblib.load(scaler_path)

    model = load_transfer_model(
        transfer_run=transfer_run,
        pretrain_run=pretrain_run,
        device=device,
    )

    lgkp_pred, mechanism_vectors = predict_candidates(
        candidates=candidates,
        model=model,
        scaler=scaler,
        device=device,
        batch_size=args.batch_size,
    )

    candidates["lgkp_pred"] = lgkp_pred

    candidate_vectors = export_candidate_vectors(
        candidates=candidates,
        mechanism_vectors=mechanism_vectors,
    )

    candidates.to_csv(
        output_dir / "all_screened_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_vectors.to_csv(
        output_dir / "all_candidate_predicted_mechanism_vectors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 3. Cross-source TRAIN-reference distance calibration
    # ------------------------------------------------------------------
    spaces, calibration, calibration_summary = build_reference_calibration(
        reference_root=reference_root,
    )
    calibration_summary.to_csv(
        output_dir / "mechanism_distance_calibration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 4. Kinetic pre-screen from all unordered pairs
    # ------------------------------------------------------------------
    all_delta = np.abs(
        lgkp_pred[pair_i_all] - lgkp_pred[pair_j_all]
    )

    prescreen_mask = all_delta <= KINETIC_PRESCREEN_TOLERANCE
    pair_i = pair_i_all[prescreen_mask]
    pair_j = pair_j_all[prescreen_mask]
    delta = all_delta[prescreen_mask]

    n_prescreen = int(len(pair_i))
    n_tight = int(np.sum(delta <= TIGHT_KINETIC_TOLERANCE))

    print(
        f"Pairs with |Delta lgkp| <= {KINETIC_PRESCREEN_TOLERANCE:.2f}: "
        f"{n_prescreen:,}"
    )
    print(
        f"Pairs with |Delta lgkp| <= {TIGHT_KINETIC_TOLERANCE:.2f}: "
        f"{n_tight:,}"
    )

    # ------------------------------------------------------------------
    # 5. D_mech for kinetically pre-screened pairs
    # ------------------------------------------------------------------
    mechanism_scores = score_mechanism_divergence(
        pair_i=pair_i,
        pair_j=pair_j,
        mechanism_vectors=mechanism_vectors,
        calibration=calibration,
    )

    pairs = build_screened_pair_table(
        candidates=candidates,
        pair_i=pair_i,
        pair_j=pair_j,
        delta_lgkp=delta,
        scores=mechanism_scores,
    )

    target = pairs.loc[
        pairs["in_final_target_region"]
    ].copy().reset_index(drop=True)

    n_target = int(len(target))

    pairs.to_csv(
        output_dir / "kinetically_matched_pair_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    target.to_csv(
        output_dir / "final_target_region_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    landscape_columns = [
        "candidate_a",
        "candidate_b",
        "lgkp_a",
        "lgkp_b",
        "abs_delta_lgkp",
        "D_mech",
        "N_eff",
        "q_mean",
        "passes_tight_kinetic_criterion",
        "passes_D_mech_threshold",
        "in_final_target_region",
    ]
    pairs[landscape_columns].to_csv(
        output_dir / "fig7b_pair_landscape.csv",
        index=False,
        encoding="utf-8-sig",
    )

    funnel = build_funnel_counts(
        n_candidates=n_candidates,
        n_all_pairs=n_all_pairs,
        n_prescreen=n_prescreen,
        n_tight=n_tight,
        n_target=n_target,
    )
    funnel.to_csv(
        output_dir / "fig7a_funnel_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 6. Manuscript representative pair
    # ------------------------------------------------------------------
    candidate_a = find_candidate_by_composition(
        candidates,
        REPRESENTATIVE_A,
        "Candidate A",
    )
    candidate_b = find_candidate_by_composition(
        candidates,
        REPRESENTATIVE_B,
        "Candidate B",
    )

    representative_pair = find_representative_pair_row(
        pairs=pairs,
        candidate_a=str(candidate_a["candidate_id"]),
        candidate_b=str(candidate_b["candidate_id"]),
    )

    composition_table = build_representative_composition_table(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )
    composition_table.to_csv(
        output_dir / "fig7c_representative_pair_composition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    channel_table = build_representative_channel_table(
        representative_pair=representative_pair,
    )
    channel_table.to_csv(
        output_dir / "fig7d_representative_pair_channel_divergence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    representative_payload = {
        "candidate_a": composition_table.iloc[0].to_dict(),
        "candidate_b": composition_table.iloc[1].to_dict(),
        "pair_metrics": representative_pair.to_dict(),
        "selection_note": (
            "This pair is the manuscript illustrative pair. It is verified to "
            "belong to the final target region; no additional ranking criterion "
            "is applied by the paper-release script."
        ),
    }
    save_json(
        representative_payload,
        output_dir / "representative_pair.json",
    )

    # ------------------------------------------------------------------
    # 7. Table S5 / Fig. S10 source data for Candidate A and Candidate B
    # ------------------------------------------------------------------
    representative_umaps = fit_representative_umaps(spaces)

    retrieval, overlays, reference_coordinates = (
        build_representative_retrieval_and_umap(
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            mechanism_vectors=mechanism_vectors,
            spaces=spaces,
            umaps=representative_umaps,
        )
    )

    retrieval.to_csv(
        output_dir / "tableS5_top5_training_reference_retrieval.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overlays.to_csv(
        output_dir / "figS10_candidate_overlays.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reference_coordinates.to_csv(
        output_dir / "figS10_umap_training_reference_coordinates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_representative_umap_preview(
        reference_coordinates=reference_coordinates,
        overlays=overlays,
        output_dir=output_dir,
    )

    make_pair_landscape_preview(
        pairs=pairs,
        representative_pair=representative_pair,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # 8. Machine-readable summary
    # ------------------------------------------------------------------
    summary = {
        "candidate_generation": {
            "initial_lhs_samples": N_LHS_SAMPLES,
            "random_state": RANDOM_STATE,
            "composition_step_at_percent": COMPOSITION_STEP,
            "search_ranges_at_percent": SEARCH_RANGES,
            "balance_element": BALANCE_ELEMENT,
            "Co_feasibility_range_at_percent": [MIN_CO, MAX_CO],
            "n_feasible_unique_candidates": n_candidates,
            "fixed_conditions": {
                "solution_temperature_C": 1200.0,
                "solution_time_h": 12.0,
                "aging_temperature_C": 900.0,
                "aging_time_h": 100.0,
                "oxidation_temperature_C": 900.0,
                "oxidation_time_h": 100.0,
            },
        },
        "screening_counts": {
            "all_unordered_pairs": n_all_pairs,
            "abs_delta_lgkp_le_0.10": n_prescreen,
            "abs_delta_lgkp_le_0.05": n_tight,
            "final_target_region": n_target,
        },
        "mechanism_divergence": {
            "channel_distance": "1 - cosine_similarity",
            "q_c_calibration": (
                "empirical percentile within cross-source TRAIN-reference "
                "pairwise distance distribution"
            ),
            "N_eff": "(sum(q_c)^2) / sum(q_c^2)",
            "D_mech": "mean(q_c) * sqrt(N_eff / 6)",
            "target_region": (
                "abs_delta_lgkp <= 0.05 and D_mech >= 0.60"
            ),
            "additional_pair_selection_rules": False,
            "q_c_0.75_role": "visualization only",
        },
        "representative_pair": {
            "candidate_a_id": str(candidate_a["candidate_id"]),
            "candidate_b_id": str(candidate_b["candidate_id"]),
            "lgkp_a": float(candidate_a["lgkp_pred"]),
            "lgkp_b": float(candidate_b["lgkp_pred"]),
            "abs_delta_lgkp": float(
                representative_pair["abs_delta_lgkp"]
            ),
            "D_mech": float(representative_pair["D_mech"]),
            "N_eff": float(representative_pair["N_eff"]),
        },
    }

    save_json(
        summary,
        output_dir / "screening_summary.json",
    )

    print("-" * 100)
    print(f"Final target-region pairs: {n_target:,}")
    print(
        "Representative pair:        "
        f"{candidate_a['candidate_id']} vs {candidate_b['candidate_id']} | "
        f"Delta lgkp={representative_pair['abs_delta_lgkp']:.6f} | "
        f"D_mech={representative_pair['D_mech']:.4f} | "
        f"N_eff={representative_pair['N_eff']:.4f}"
    )
    print(f"Outputs:                    {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
