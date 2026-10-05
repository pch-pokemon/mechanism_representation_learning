#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mechanism-space analysis for selected held-out oxidation-kinetics cases.
----------------------------
This script reproduces the quantitative and retrieval data underlying the
held-out mechanism-interpretation cases shown in Fig. 6 and Fig. S9.

Scientific separation enforced by this script
---------------------------------------------
1. Query vectors are the six mechanism representations predicted for a held-out
   TEST sample by the frozen-transfer model.

2. Mechanism interpretation uses ONLY the TRAINING-reference library:
       predicted vector -> Top-5 nearest training-reference mechanisms

3. The held-out sample's original literature-derived mechanism representation
   and mechanism text are NEVER included in the retrieval library.

4. The held-out original representation is accessed only after retrieval, for
   post-hoc validation:
       CS(P,O) = cosine(predicted vector, held-out original vector)

5. UMAP is fitted independently for each mechanism channel using TRAINING
   references only. Predicted and held-out original vectors are transformed
   with the fitted mapping without refitting.

6. No LLM is called here. Retrieval evidence and held-out original mechanisms
   are exported to separate files so that interpretation can be generated from
   the retrieval evidence before post-hoc comparison with the original
   literature description.

Inputs
------
A. Final held-out predictions produced by evaluate_kinetics.py:
   <test_evaluation>/fig5b_test_predictions.csv

B. Predicted six-channel mechanism vectors produced by evaluate_kinetics.py:
   <test_evaluation>/test_predicted_mechanism_vectors.csv

C. Channel-specific reference spaces produced by build_mechanism_embeddings.py:
   <reference_root>/channel_1/reference_embeddings_64d.csv
   ...
   <reference_root>/channel_6/reference_embeddings_64d.csv

Case specification
------------------
Each case is supplied using the strict four-key identity:

    Ref_source_num|sample_id|alloy_id|condition_id

Example
-------
python src/analyze_mechanism_cases.py \
    --test-predictions outputs/frozen_transfer/test_evaluation/fig5b_test_predictions.csv \
    --test-vectors outputs/frozen_transfer/test_evaluation/test_predicted_mechanism_vectors.csv \
    --reference-root outputs/mechanism_embeddings \
    --case "rXXX|sYYY|AlloyName|900" \
    --output-dir outputs/fig6_case

For a second held-out case (e.g. Fig. S9), run the same command with a different
--case and output directory.

Outputs
-------
case_summary.csv
case_channel_metrics.csv
top5_training_reference_retrieval.csv
heldout_original_mechanisms.csv
case_mechanism_vectors.csv
umap_training_reference_coordinates.csv
umap_case_overlays.csv
interpretation_input.json
posthoc_validation.json
case_umap_preview.png / .pdf
analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "analyze_mechanism_cases.py requires umap-learn. "
        "Install it with: pip install umap-learn"
    ) from exc


# =============================================================================
# Fixed manuscript settings
# =============================================================================

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

CHANNEL_SHORT_NAMES: Dict[int, str] = {
    1: "Protective scale",
    2: "Internal/selective oxidation",
    3: "Non-protective/transient oxides",
    4: "Transport/defect kinetics",
    5: "Scale failure/spallation",
    6: "Alloying-element interaction",
}

CHANNELS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
EMBED_DIM = 64
TOP_K = 5

UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.15
UMAP_METRIC = "cosine"
RANDOM_STATE = 42


# =============================================================================
# Generic helpers
# =============================================================================

def clean_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""

    # Keep identifiers stable while normalizing simple numeric-like values.
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer() and "." in text:
            return str(int(number))
    except (TypeError, ValueError):
        pass

    return text


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Ref_source_num" not in out.columns and "ref_source_num" in out.columns:
        out = out.rename(columns={"ref_source_num": "Ref_source_num"})

    missing = [key for key in ALIGN_KEYS if key not in out.columns]
    if missing:
        raise ValueError(f"Missing alignment-key columns: {missing}")

    for key in ALIGN_KEYS:
        out[key] = out[key].map(clean_key)

    return out


def validate_unique_keys(df: pd.DataFrame, label: str) -> None:
    empty = df[list(ALIGN_KEYS)].eq("").any(axis=1)
    if empty.any():
        examples = df.loc[empty, list(ALIGN_KEYS)].head(20)
        raise ValueError(f"{label} contains empty alignment keys:\n{examples}")

    duplicated = df.duplicated(list(ALIGN_KEYS), keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, list(ALIGN_KEYS)].head(20)
        raise ValueError(f"{label} contains duplicate alignment keys:\n{examples}")


def key_tuple_from_mapping(
    mapping: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    return tuple(
        clean_key(mapping.get(key, ""))
        for key in ALIGN_KEYS
    )  # type: ignore[return-value]


def row_key_tuple(
    row: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    return tuple(
        clean_key(row[key])
        for key in ALIGN_KEYS
    )  # type: ignore[return-value]


def l2_normalize(
    array: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("A zero-norm mechanism vector was encountered.")

    return x / np.clip(norms, eps, None)


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a_norm = l2_normalize(a)[0]
    b_norm = l2_normalize(b)[0]
    return float(np.dot(a_norm, b_norm))


def save_json(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): convert(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
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


def parse_case(value: str) -> Dict[str, str]:
    parts = [part.strip() for part in value.split("|")]

    if len(parts) != 4:
        raise ValueError(
            "--case must contain exactly four pipe-separated fields: "
            "Ref_source_num|sample_id|alloy_id|condition_id"
        )

    case = dict(zip(ALIGN_KEYS, parts))

    if any(clean_key(case[key]) == "" for key in ALIGN_KEYS):
        raise ValueError("--case contains an empty alignment-key field.")

    return case


# =============================================================================
# Held-out prediction inputs
# =============================================================================

def load_test_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test prediction file not found: {path}")

    df = pd.read_csv(path)
    df = normalize_keys(df)
    validate_unique_keys(df, path.name)

    required = {"lgkp_true", "lgkp_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}"
        )

    if "split" in df.columns:
        split = (
            df["split"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        invalid = ~split.eq("test")
        if invalid.any():
            raise ValueError(
                "fig5b_test_predictions.csv must contain held-out TEST rows only."
            )

    return df


def load_test_vectors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test mechanism-vector file not found: {path}")

    df = pd.read_csv(path)
    df = normalize_keys(df)

    required = {"channel", "channel_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}"
        )

    df["channel"] = pd.to_numeric(
        df["channel"],
        errors="raise",
    ).astype(int)

    vector_columns = [f"emb_{i}" for i in range(EMBED_DIM)]
    missing_vectors = [
        column
        for column in vector_columns
        if column not in df.columns
    ]
    if missing_vectors:
        raise ValueError(
            f"{path.name} is missing 64-D mechanism columns: "
            f"{missing_vectors[:5]}"
        )

    duplicated = df.duplicated(
        list(ALIGN_KEYS) + ["channel"],
        keep=False,
    )
    if duplicated.any():
        examples = df.loc[
            duplicated,
            list(ALIGN_KEYS) + ["channel"],
        ].head(20)
        raise ValueError(
            "Predicted mechanism-vector table contains duplicate sample-channel rows:\n"
            f"{examples}"
        )

    channels = set(df["channel"].unique())
    missing_channels = sorted(set(CHANNELS) - channels)
    if missing_channels:
        raise ValueError(
            f"Predicted vector table is missing channels: {missing_channels}"
        )

    return df


def get_selected_case(
    predictions: pd.DataFrame,
    case: Mapping[str, Any],
) -> pd.Series:
    key = key_tuple_from_mapping(case)

    mask = np.ones(len(predictions), dtype=bool)
    for column, value in zip(ALIGN_KEYS, key):
        mask &= predictions[column].astype(str).eq(value).to_numpy()

    matched = predictions.loc[mask]

    if len(matched) == 0:
        raise ValueError(
            f"Selected case is not present in held-out test predictions: {key}"
        )
    if len(matched) > 1:
        raise RuntimeError(
            f"Selected case matched multiple test rows: {key}"
        )

    return matched.iloc[0]


def get_predicted_vector(
    vectors: pd.DataFrame,
    case_key: Tuple[str, str, str, str],
    channel: int,
) -> np.ndarray:
    mask = vectors["channel"].eq(channel).to_numpy()

    for column, value in zip(ALIGN_KEYS, case_key):
        mask &= vectors[column].astype(str).eq(value).to_numpy()

    matched = vectors.loc[mask]

    if len(matched) != 1:
        raise ValueError(
            f"Expected one predicted vector for case={case_key}, "
            f"channel={channel}; found {len(matched)}."
        )

    vector = matched[
        [f"emb_{i}" for i in range(EMBED_DIM)]
    ].to_numpy(dtype=np.float32)[0]

    if not np.isfinite(vector).all():
        raise ValueError(
            f"Predicted vector contains NaN/inf for channel {channel}."
        )

    return vector


# =============================================================================
# Reference mechanism spaces
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
            f"Channel-{channel} reference space not found: {path}"
        )

    df = pd.read_csv(path)
    df = normalize_keys(df)
    validate_unique_keys(df, f"channel {channel} reference space")

    required = {
        "split",
        "embedding_valid",
        "mechanism_text",
        "canonical_tags",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Channel-{channel} reference CSV is missing columns: {missing}"
        )

    vector_columns = [f"emb_{i}" for i in range(EMBED_DIM)]
    missing_vectors = [
        column
        for column in vector_columns
        if column not in df.columns
    ]
    if missing_vectors:
        raise ValueError(
            f"Channel-{channel} reference CSV does not contain "
            f"all {EMBED_DIM} embedding columns."
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

    if not df["embedding_valid"].isin([0, 1]).all():
        raise ValueError(
            f"Channel-{channel} embedding_valid must contain only 0/1."
        )

    matrix = df[vector_columns].to_numpy(dtype=np.float32)

    valid = df["embedding_valid"].eq(1).to_numpy()
    if not np.isfinite(matrix[valid]).all():
        raise ValueError(
            f"Channel-{channel} has NaN/inf among valid reference vectors."
        )

    train_mask = valid & df["split"].eq("train").to_numpy()

    library = df.loc[train_mask].copy().reset_index(drop=True)
    library_matrix = library[vector_columns].to_numpy(dtype=np.float32)

    if len(library) < TOP_K:
        raise ValueError(
            f"Channel-{channel} has only {len(library)} valid training references."
        )

    return {
        "path": path,
        "full": df,
        "vector_columns": vector_columns,
        "library": library,
        "library_matrix": library_matrix,
        "library_matrix_normalized": l2_normalize(library_matrix),
    }


def load_reference_spaces(
    reference_root: Path,
) -> Dict[int, Dict[str, Any]]:
    spaces: Dict[int, Dict[str, Any]] = {}

    for channel in CHANNELS:
        spaces[channel] = load_reference_space(
            reference_root=reference_root,
            channel=channel,
        )

        full = spaces[channel]["full"]
        library = spaces[channel]["library"]

        print(
            f"Channel {channel}: "
            f"total={len(full)}, "
            f"reported={int(full['embedding_valid'].sum())}, "
            f"training library={len(library)}"
        )

    return spaces


def find_original_reference(
    space: Mapping[str, Any],
    case_key: Tuple[str, str, str, str],
) -> pd.Series:
    full: pd.DataFrame = space["full"]

    mask = np.ones(len(full), dtype=bool)
    for column, value in zip(ALIGN_KEYS, case_key):
        mask &= full[column].astype(str).eq(value).to_numpy()

    matched = full.loc[mask]

    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one original reference row for {case_key}; "
            f"found {len(matched)}."
        )

    row = matched.iloc[0]

    if int(row["embedding_valid"]) != 1:
        raise ValueError(
            f"Selected case has no reported original mechanism for "
            f"channel {row.get('channel', '')}."
        )

    # Strict leakage guard.
    if str(row["split"]).strip().lower() == "train":
        raise ValueError(
            "Selected held-out case is assigned to the TRAIN reference library. "
            "This violates the case-analysis separation rule."
        )

    return row


# =============================================================================
# Retrieval and UMAP
# =============================================================================

def retrieve_top5(
    predicted_vector: np.ndarray,
    space: Mapping[str, Any],
    case_key: Tuple[str, str, str, str],
) -> pd.DataFrame:
    """
    Retrieve exactly five nearest TRAINING-reference mechanism states.
    """
    library: pd.DataFrame = space["library"]
    library_norm: np.ndarray = space["library_matrix_normalized"]
    query = l2_normalize(predicted_vector)[0]

    # Exact sample self-exclusion is retained as an additional safety guard.
    exact_self = np.ones(len(library), dtype=bool)
    for column, value in zip(ALIGN_KEYS, case_key):
        exact_self &= library[column].astype(str).eq(value).to_numpy()

    if exact_self.any():
        raise RuntimeError(
            "The selected held-out case unexpectedly appears in the "
            "training-reference library."
        )

    similarities = library_norm @ query
    order = np.argsort(-similarities)[:TOP_K]

    top = library.iloc[order].copy().reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, TOP_K + 1, dtype=int))
    top.insert(
        1,
        "cosine_similarity",
        similarities[order].astype(float),
    )
    top["library_index"] = order.astype(int)

    return top


def fit_umaps(
    spaces: Mapping[int, Mapping[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    for channel in CHANNELS:
        library: pd.DataFrame = spaces[channel]["library"]
        matrix: np.ndarray = spaces[channel]["library_matrix"]

        n_neighbors = min(
            UMAP_N_NEIGHBORS,
            len(library) - 1,
        )
        if n_neighbors < 2:
            raise ValueError(
                f"Channel-{channel} has too few training references for UMAP."
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
            matrix
        ).astype(np.float32)

        results[channel] = {
            "reducer": reducer,
            "library_coordinates": coordinates,
        }

    return results


# =============================================================================
# Selected-case analysis
# =============================================================================

def analyze_case(
    case_row: pd.Series,
    predicted_vectors: pd.DataFrame,
    spaces: Mapping[int, Mapping[str, Any]],
    umaps: Mapping[int, Mapping[str, Any]],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
    Dict[str, Any],
]:
    case_key = row_key_tuple(case_row)

    metric_rows: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []
    original_rows: List[Dict[str, Any]] = []
    vector_rows: List[Dict[str, Any]] = []
    overlay_rows: List[Dict[str, Any]] = []
    reference_coord_frames: List[pd.DataFrame] = []

    interpretation_channels: Dict[str, Any] = {}
    validation_channels: Dict[str, Any] = {}

    for channel in CHANNELS:
        space = spaces[channel]
        reducer = umaps[channel]["reducer"]

        predicted = get_predicted_vector(
            vectors=predicted_vectors,
            case_key=case_key,
            channel=channel,
        )

        original_row = find_original_reference(
            space=space,
            case_key=case_key,
        )
        original = original_row[
            space["vector_columns"]
        ].to_numpy(dtype=np.float32)

        cs_po = cosine_similarity(
            predicted,
            original,
        )

        top = retrieve_top5(
            predicted_vector=predicted,
            space=space,
            case_key=case_key,
        )

        pred_xy = reducer.transform(
            predicted.reshape(1, -1)
        )[0]
        original_xy = reducer.transform(
            original.reshape(1, -1)
        )[0]

        # ------------------------------------------------------------------
        # Channel-level quantitative metrics
        # ------------------------------------------------------------------
        metric_rows.append(
            {
                **{
                    key: case_key[index]
                    for index, key in enumerate(ALIGN_KEYS)
                },
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "CS_P_O": cs_po,
                "top1_CS_P_R": float(top.iloc[0]["cosine_similarity"]),
                "mean_top5_CS_P_R": float(top["cosine_similarity"].mean()),
                "min_top5_CS_P_R": float(top["cosine_similarity"].min()),
                "top5_unique_sources": int(
                    top["Ref_source_num"].nunique()
                ),
                "training_reference_library_size": int(
                    len(space["library"])
                ),
            }
        )

        # ------------------------------------------------------------------
        # Prediction and original vectors
        # ------------------------------------------------------------------
        for vector_type, vector in (
            ("prediction", predicted),
            ("heldout_original", original),
        ):
            row: Dict[str, Any] = {
                **{
                    key: case_key[index]
                    for index, key in enumerate(ALIGN_KEYS)
                },
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "vector_type": vector_type,
            }
            row.update(
                {
                    f"emb_{dim}": float(value)
                    for dim, value in enumerate(vector)
                }
            )
            vector_rows.append(row)

        # ------------------------------------------------------------------
        # Held-out original mechanism: post-hoc only
        # ------------------------------------------------------------------
        original_rows.append(
            {
                **{
                    key: case_key[index]
                    for index, key in enumerate(ALIGN_KEYS)
                },
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "split": str(original_row["split"]),
                "CS_P_O": cs_po,
                "canonical_tags": str(original_row["canonical_tags"]),
                "mechanism_text": str(original_row["mechanism_text"]),
                "evidence_status": str(
                    original_row.get("evidence_status", "")
                ),
            }
        )

        validation_channels[str(channel)] = {
            "channel_name": CHANNEL_NAMES[channel],
            "CS_P_O": cs_po,
            "heldout_original": {
                "Ref_source_num": case_key[0],
                "sample_id": case_key[1],
                "alloy_id": case_key[2],
                "condition_id": case_key[3],
                "canonical_tags": str(
                    original_row["canonical_tags"]
                ),
                "mechanism_text": str(
                    original_row["mechanism_text"]
                ),
            },
        }

        # ------------------------------------------------------------------
        # Top-5 training-reference retrieval
        # ------------------------------------------------------------------
        channel_retrieval_for_json: List[Dict[str, Any]] = []

        for _, neighbor in top.iterrows():
            neighbor_key = row_key_tuple(neighbor)
            library_index = int(neighbor["library_index"])
            xy = umaps[channel]["library_coordinates"][library_index]

            record = {
                "query_Ref_source_num": case_key[0],
                "query_sample_id": case_key[1],
                "query_alloy_id": case_key[2],
                "query_condition_id": case_key[3],
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "rank": int(neighbor["rank"]),
                "CS_P_R": float(neighbor["cosine_similarity"]),
                "Ref_source_num": neighbor_key[0],
                "sample_id": neighbor_key[1],
                "alloy_id": neighbor_key[2],
                "condition_id": neighbor_key[3],
                "split": str(neighbor["split"]),
                "canonical_tags": str(neighbor["canonical_tags"]),
                "mechanism_text": str(neighbor["mechanism_text"]),
                "evidence_status": str(
                    neighbor.get("evidence_status", "")
                ),
                "umap_x": float(xy[0]),
                "umap_y": float(xy[1]),
            }
            retrieval_rows.append(record)

            channel_retrieval_for_json.append(
                {
                    "rank": int(neighbor["rank"]),
                    "CS_P_R": float(
                        neighbor["cosine_similarity"]
                    ),
                    "Ref_source_num": neighbor_key[0],
                    "sample_id": neighbor_key[1],
                    "alloy_id": neighbor_key[2],
                    "condition_id": neighbor_key[3],
                    "canonical_tags": str(
                        neighbor["canonical_tags"]
                    ),
                    "mechanism_text": str(
                        neighbor["mechanism_text"]
                    ),
                }
            )

        interpretation_channels[str(channel)] = {
            "channel_name": CHANNEL_NAMES[channel],
            "retrieval_rule": (
                "Top-5 cosine-nearest mechanisms from TRAIN references only"
            ),
            "top5_training_references": channel_retrieval_for_json,
        }

        # ------------------------------------------------------------------
        # UMAP overlays
        # ------------------------------------------------------------------
        overlay_rows.append(
            {
                **{
                    key: case_key[index]
                    for index, key in enumerate(ALIGN_KEYS)
                },
                "channel": channel,
                "point_type": "prediction",
                "rank": np.nan,
                "cosine_similarity": np.nan,
                "umap_x": float(pred_xy[0]),
                "umap_y": float(pred_xy[1]),
            }
        )

        overlay_rows.append(
            {
                **{
                    key: case_key[index]
                    for index, key in enumerate(ALIGN_KEYS)
                },
                "channel": channel,
                "point_type": "heldout_original",
                "rank": np.nan,
                "cosine_similarity": cs_po,
                "umap_x": float(original_xy[0]),
                "umap_y": float(original_xy[1]),
            }
        )

        for _, neighbor in top.iterrows():
            library_index = int(neighbor["library_index"])
            xy = umaps[channel]["library_coordinates"][library_index]
            neighbor_key = row_key_tuple(neighbor)

            overlay_rows.append(
                {
                    **{
                        key: neighbor_key[index]
                        for index, key in enumerate(ALIGN_KEYS)
                    },
                    "channel": channel,
                    "point_type": "top5_training_reference",
                    "rank": int(neighbor["rank"]),
                    "cosine_similarity": float(
                        neighbor["cosine_similarity"]
                    ),
                    "umap_x": float(xy[0]),
                    "umap_y": float(xy[1]),
                }
            )

        # ------------------------------------------------------------------
        # Reusable training-reference coordinates
        # ------------------------------------------------------------------
        library = space["library"].copy().reset_index(drop=True)
        coordinates = umaps[channel]["library_coordinates"]

        frame = library[
            list(ALIGN_KEYS)
            + [
                "split",
                "canonical_tags",
                "mechanism_text",
            ]
        ].copy()

        frame.insert(0, "channel", channel)
        frame.insert(1, "channel_name", CHANNEL_NAMES[channel])
        frame["umap_x"] = coordinates[:, 0]
        frame["umap_y"] = coordinates[:, 1]
        reference_coord_frames.append(frame)

    metrics = pd.DataFrame(metric_rows)

    case_summary = pd.DataFrame(
        [
            {
                **{
                    key: case_key[index]
                    for index, key in enumerate(ALIGN_KEYS)
                },
                "lgkp_true": float(case_row["lgkp_true"]),
                "lgkp_pred": float(case_row["lgkp_pred"]),
                "abs_error": float(
                    abs(
                        float(case_row["lgkp_pred"])
                        - float(case_row["lgkp_true"])
                    )
                ),
                "mean_CS_P_O": float(metrics["CS_P_O"].mean()),
                "min_CS_P_O": float(metrics["CS_P_O"].min()),
            }
        ]
    )

    interpretation_input = {
        "case_identity": {
            key: case_key[index]
            for index, key in enumerate(ALIGN_KEYS)
        },
        "important_instruction": (
            "Generate channel-level mechanism interpretations using ONLY "
            "the Top-5 training-reference mechanisms below. Do not use "
            "held-out original mechanisms during interpretation generation."
        ),
        "channels": interpretation_channels,
    }

    posthoc_validation = {
        "case_identity": {
            key: case_key[index]
            for index, key in enumerate(ALIGN_KEYS)
        },
        "important_instruction": (
            "These held-out original literature mechanisms are reserved "
            "for post-hoc validation after prediction-side interpretation."
        ),
        "channels": validation_channels,
    }

    return (
        case_summary,
        metrics,
        pd.DataFrame(retrieval_rows),
        pd.DataFrame(original_rows),
        pd.DataFrame(vector_rows),
        pd.DataFrame(overlay_rows),
        pd.concat(reference_coord_frames, ignore_index=True),
        interpretation_input,
        posthoc_validation,
    )


# =============================================================================
# Preview plot
# =============================================================================

def plot_case_umap_preview(
    reference_coordinates: pd.DataFrame,
    overlays: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Compact 2 x 3 preview. The CSV files are the primary source data for the
    publication figure; this plot is provided as a reproducibility check.
    """
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.0, 7.2),
    )
    axes = axes.ravel()

    for axis, channel in zip(axes, CHANNELS):
        background = reference_coordinates.loc[
            reference_coordinates["channel"].eq(channel)
        ]

        overlay = overlays.loc[
            overlays["channel"].eq(channel)
        ]

        axis.scatter(
            background["umap_x"],
            background["umap_y"],
            s=10,
            alpha=0.25,
            linewidths=0,
        )

        neighbours = overlay.loc[
            overlay["point_type"].eq(
                "top5_training_reference"
            )
        ]
        if not neighbours.empty:
            axis.scatter(
                neighbours["umap_x"],
                neighbours["umap_y"],
                s=45,
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

        original = overlay.loc[
            overlay["point_type"].eq("heldout_original")
        ]
        if not original.empty:
            axis.scatter(
                original["umap_x"],
                original["umap_y"],
                s=65,
                marker="s",
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

        axis.set_title(
            f"Ch{channel}: {CHANNEL_SHORT_NAMES[channel]}"
        )
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()

    fig.savefig(
        output_dir / "case_umap_preview.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "case_umap_preview.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one selected held-out test case using "
            "training-reference mechanism neighbourhoods."
        )
    )

    parser.add_argument(
        "--test-predictions",
        required=True,
        type=Path,
        help="fig5b_test_predictions.csv from evaluate_kinetics.py.",
    )
    parser.add_argument(
        "--test-vectors",
        required=True,
        type=Path,
        help=(
            "test_predicted_mechanism_vectors.csv "
            "from evaluate_kinetics.py."
        ),
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        type=Path,
        help="Root produced by build_mechanism_embeddings.py.",
    )
    parser.add_argument(
        "--case",
        required=True,
        type=str,
        help=(
            "Held-out case as "
            "Ref_source_num|sample_id|alloy_id|condition_id."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case = parse_case(args.case)

    predictions = load_test_predictions(
        args.test_predictions
    )
    predicted_vectors = load_test_vectors(
        args.test_vectors
    )

    selected_case = get_selected_case(
        predictions=predictions,
        case=case,
    )

    case_key = row_key_tuple(selected_case)

    print("=" * 96)
    print("Held-out mechanism-space case analysis")
    print("=" * 96)
    print(
        "Case: "
        + " | ".join(case_key)
    )
    print(
        f"Experimental lgkp: {float(selected_case['lgkp_true']):.6f}"
    )
    print(
        f"Predicted lgkp:    {float(selected_case['lgkp_pred']):.6f}"
    )
    print("Retrieval library: TRAIN references only")
    print("Top-k:             5 (fixed)")
    print("=" * 96)

    spaces = load_reference_spaces(
        args.reference_root
    )
    umaps = fit_umaps(spaces)

    (
        case_summary,
        metrics,
        top5,
        originals,
        vectors,
        overlays,
        reference_coordinates,
        interpretation_input,
        posthoc_validation,
    ) = analyze_case(
        case_row=selected_case,
        predicted_vectors=predicted_vectors,
        spaces=spaces,
        umaps=umaps,
    )

    # ------------------------------------------------------------------
    # Export source data
    # ------------------------------------------------------------------
    case_summary.to_csv(
        output_dir / "case_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        output_dir / "case_channel_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top5.to_csv(
        output_dir / "top5_training_reference_retrieval.csv",
        index=False,
        encoding="utf-8-sig",
    )

    originals.to_csv(
        output_dir / "heldout_original_mechanisms.csv",
        index=False,
        encoding="utf-8-sig",
    )

    vectors.to_csv(
        output_dir / "case_mechanism_vectors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overlays.to_csv(
        output_dir / "umap_case_overlays.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reference_coordinates.to_csv(
        output_dir / "umap_training_reference_coordinates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        interpretation_input,
        output_dir / "interpretation_input.json",
    )

    save_json(
        posthoc_validation,
        output_dir / "posthoc_validation.json",
    )

    plot_case_umap_preview(
        reference_coordinates=reference_coordinates,
        overlays=overlays,
        output_dir=output_dir,
    )

    analysis_summary = {
        "case_identity": {
            key: case_key[index]
            for index, key in enumerate(ALIGN_KEYS)
        },
        "lgkp_true": float(selected_case["lgkp_true"]),
        "lgkp_pred": float(selected_case["lgkp_pred"]),
        "abs_error": float(
            abs(
                float(selected_case["lgkp_pred"])
                - float(selected_case["lgkp_true"])
            )
        ),
        "reference_library_split": "train",
        "top_k": TOP_K,
        "embedding_dimension": EMBED_DIM,
        "mean_CS_P_O": float(
            metrics["CS_P_O"].mean()
        ),
        "channel_CS_P_O": {
            str(int(row["channel"])): float(row["CS_P_O"])
            for _, row in metrics.iterrows()
        },
        "umap": {
            "fit_data": "training-reference mechanisms only",
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "random_state": RANDOM_STATE,
            "predicted_and_original_vectors": (
                "transformed with fitted mapping without refitting"
            ),
        },
        "interpretation_original_separation": (
            "Top-5 retrieval evidence and held-out original mechanisms "
            "are exported separately."
        ),
    }

    save_json(
        analysis_summary,
        output_dir / "analysis_summary.json",
    )

    print("-" * 96)
    for _, row in metrics.iterrows():
        print(
            f"Ch{int(row['channel'])}: "
            f"CS(P,O)={row['CS_P_O']:.4f}, "
            f"Top-1 CS(P,R)={row['top1_CS_P_R']:.4f}, "
            f"mean Top-5={row['mean_top5_CS_P_R']:.4f}"
        )

    print("-" * 96)
    print(
        f"Mean CS(P,O): {metrics['CS_P_O'].mean():.4f}"
    )
    print(f"Outputs:      {output_dir}")
    print("=" * 96)


if __name__ == "__main__":
    main()
