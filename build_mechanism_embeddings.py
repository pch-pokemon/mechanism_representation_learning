# -*- coding: utf-8 -*-

"""
Build six channel-specific literature-derived mechanism reference spaces.

Manuscript workflow
-------------------
1. Read the joint six-channel mechanism annotations.
2. Use ONLY the textual mechanism description as semantic input.
3. Encode reported mechanism descriptions with fixed pretrained SciBERT.
4. Mean-pool final-layer token representations and apply L2 normalization.
5. Fit an independent PCA model for each mechanism channel using TRAINING
   embeddings only.
6. Transform training, validation and test embeddings using the same fitted PCA.
7. Export one 64-dimensional reference representation per reported
   sample-channel pair.

Canonical tags are retained only as metadata and are never included in the
semantic embedding input.

Rows labelled "not_reported" remain aligned in the output but are assigned
zero-vector placeholders and excluded from PCA fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Fixed manuscript settings
# =============================================================================

SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"

CHANNEL_NAMES: Dict[int, str] = {
    1: "protective_scale_formation",
    2: "internal_oxidation_and_selective_oxidation",
    3: "nonprotective_or_transient_oxide_formation",
    4: "transport_and_defect_kinetics",
    5: "scale_failure_and_spallation",
    6: "alloying_element_interaction",
}

ALIGN_KEYS = [
    "Ref_source_num",
    "sample_id",
    "alloy_id",
    "condition_id",
]

EMBEDDING_DIM = 64

# Used only to reproduce Table S2.
PCA_REPORT_DIMS = (16, 32, 64, 128)

MAX_LENGTH = 256
RANDOM_STATE = 42


# =============================================================================
# Utilities
# =============================================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def clean_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    text = clean_text(value)
    return [text] if text else []


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ALIGN_KEYS:
        if column not in df.columns:
            raise ValueError(f"Missing identifier column: {column}")

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if df[column].eq("").any():
            raise ValueError(
                f"Empty values found in identifier column: {column}"
            )

    return df


# =============================================================================
# Fixed split manifest
# =============================================================================

def load_split_manifest(path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(path, dtype=str)
    split_df = normalize_ids(split_df)

    if "split" not in split_df.columns:
        raise ValueError(
            "Split manifest must contain a 'split' column."
        )

    split_df["split"] = (
        split_df["split"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {"train", "val", "test"}

    invalid = ~split_df["split"].isin(allowed)
    if invalid.any():
        examples = split_df.loc[
            invalid,
            ALIGN_KEYS + ["split"]
        ].head()

        raise ValueError(
            "Split values must be train/val/test.\n"
            f"Examples:\n{examples}"
        )

    if split_df.duplicated(ALIGN_KEYS).any():
        raise ValueError(
            "Duplicate sample-condition identities found in split manifest."
        )

    return split_df[ALIGN_KEYS + ["split"]].copy()


# =============================================================================
# Joint six-channel mechanism JSON
# =============================================================================

def normalize_json_blocks(data: Any) -> List[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        return [data]

    if isinstance(data, list):
        blocks = [
            item for item in data
            if isinstance(item, Mapping)
        ]
        if not blocks:
            raise ValueError("No valid JSON blocks were found.")
        return blocks

    raise ValueError(
        "Mechanism JSON root must be an object or a list of objects."
    )


def load_channel_records(
    json_path: Path,
    channel_id: int,
) -> pd.DataFrame:
    """
    Read one channel from the joint six-channel mechanism JSON.

    Expected schema
    ---------------
    {
        "Ref_source_num": "...",
        "target_samples": [
            {
                "sample_id": "...",
                "alloy_id": "...",
                "condition_id": "...",
                "channels": {
                    "<channel_name>": {
                        "tags": [...],
                        "text": "...",
                        "status": "reported | not_reported"
                    }
                }
            }
        ]
    }
    """

    channel_name = CHANNEL_NAMES[channel_id]

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    blocks = normalize_json_blocks(data)
    rows: List[Dict[str, Any]] = []

    for block in blocks:
        block_ref = clean_text(
            block.get("Ref_source_num", "")
        )

        samples = block.get("target_samples", [])

        if not isinstance(samples, list):
            raise ValueError(
                "'target_samples' must be a list."
            )

        for sample in samples:
            if not isinstance(sample, Mapping):
                continue

            channels = sample.get("channels", {})
            if not isinstance(channels, Mapping):
                channels = {}

            payload = channels.get(channel_name, {})
            if not isinstance(payload, Mapping):
                payload = {}

            status = clean_text(
                payload.get("status", "not_reported")
            ).lower()

            if not status:
                status = "not_reported"

            mechanism_text = clean_text(
                payload.get("text", "")
            )

            tags = clean_list(
                payload.get("tags", [])
            )

            reported = status != "not_reported"

            if reported and not mechanism_text:
                raise ValueError(
                    "A reported mechanism record has empty mechanism text: "
                    f"{sample.get('sample_id', '')}, "
                    f"channel={channel_id}"
                )

            rows.append(
                {
                    "Ref_source_num": clean_text(
                        sample.get(
                            "Ref_source_num",
                            block_ref,
                        )
                    ),
                    "sample_id": clean_text(
                        sample.get("sample_id", "")
                    ),
                    "alloy_id": clean_text(
                        sample.get("alloy_id", "")
                    ),
                    "condition_id": clean_text(
                        sample.get("condition_id", "")
                    ),
                    "channel": channel_id,
                    "channel_name": channel_name,
                    "mechanism_text": mechanism_text,
                    "canonical_tags": json.dumps(
                        tags,
                        ensure_ascii=False,
                    ),
                    "evidence_status": status,
                    "embedding_valid": int(reported),
                }
            )

    if not rows:
        raise ValueError(
            f"No mechanism records found for channel {channel_id}."
        )

    df = pd.DataFrame(rows)
    df = normalize_ids(df)

    if df.duplicated(ALIGN_KEYS).any():
        duplicated = df.loc[
            df.duplicated(
                ALIGN_KEYS,
                keep=False,
            ),
            ALIGN_KEYS,
        ].head(20)

        raise ValueError(
            "Duplicate sample-condition identities found in mechanism JSON.\n"
            f"{duplicated}"
        )

    return df


def attach_split(
    mechanism_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:

    merged = mechanism_df.merge(
        split_df,
        on=ALIGN_KEYS,
        how="left",
        validate="one_to_one",
    )

    if merged["split"].isna().any():
        missing = merged.loc[
            merged["split"].isna(),
            ALIGN_KEYS,
        ].head(20)

        raise ValueError(
            "Some mechanism records do not have a fixed split assignment.\n"
            f"{missing}"
        )

    return merged


# =============================================================================
# SciBERT encoding
# =============================================================================

def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print(
                "CUDA is unavailable; using CPU instead."
            )
            return torch.device("cpu")

    return torch.device(requested)


def encode_scibert(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    SciBERT final-layer mean pooling followed by L2 normalization.
    """

    vectors: List[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(
                texts[start:start + batch_size]
            )

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            output = model(**encoded)

            token_embeddings = output.last_hidden_state
            attention_mask = encoded[
                "attention_mask"
            ]

            expanded_mask = (
                attention_mask
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )

            summed = (
                token_embeddings
                * expanded_mask
            ).sum(dim=1)

            denominator = (
                expanded_mask
                .sum(dim=1)
                .clamp(min=1e-9)
            )

            sentence_embeddings = (
                summed / denominator
            )

            # Manuscript: L2-normalized semantic embeddings.
            sentence_embeddings = F.normalize(
                sentence_embeddings,
                p=2,
                dim=1,
            )

            vectors.append(
                sentence_embeddings
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    result = np.vstack(vectors)

    if not np.isfinite(result).all():
        raise RuntimeError(
            "SciBERT embeddings contain non-finite values."
        )

    return result


# =============================================================================
# Channel-specific PCA
# =============================================================================

def calculate_explained_variance(
    train_embeddings: np.ndarray,
) -> Dict[int, float]:

    max_dim = max(PCA_REPORT_DIMS)

    max_components = min(
        train_embeddings.shape[0],
        train_embeddings.shape[1],
    )

    if max_components < max_dim:
        raise ValueError(
            f"Training set permits only {max_components} PCA components; "
            f"{max_dim} are required to reproduce Table S2."
        )

    diagnostic_pca = PCA(
        n_components=max_dim,
        random_state=RANDOM_STATE,
    )

    diagnostic_pca.fit(train_embeddings)

    cumulative = np.cumsum(
        diagnostic_pca.explained_variance_ratio_
    )

    return {
        dim: float(cumulative[dim - 1])
        for dim in PCA_REPORT_DIMS
    }


def build_channel_reference_space(
    channel_id: int,
    mechanism_json: Path,
    split_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> Dict[str, Any]:

    channel_name = CHANNEL_NAMES[channel_id]

    print("=" * 80)
    print(
        f"Channel {channel_id}: {channel_name}"
    )
    print("=" * 80)

    df = load_channel_records(
        mechanism_json,
        channel_id,
    )

    df = attach_split(
        df,
        split_df,
    )

    valid_mask = (
        df["embedding_valid"]
        .eq(1)
        .to_numpy()
    )

    train_mask = (
        valid_mask
        & df["split"]
        .eq("train")
        .to_numpy()
    )

    if train_mask.sum() <= EMBEDDING_DIM:
        raise ValueError(
            f"Channel {channel_id} has only "
            f"{train_mask.sum()} reported training records; "
            f"cannot fit {EMBEDDING_DIM}-D PCA."
        )

    valid_texts = (
        df.loc[
            valid_mask,
            "mechanism_text",
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"Total aligned records:      {len(df)}"
    )
    print(
        f"Reported records:           {valid_mask.sum()}"
    )
    print(
        f"Reported training records:  {train_mask.sum()}"
    )

    raw_valid = encode_scibert(
        valid_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
    )

    raw_dim = raw_valid.shape[1]

    raw_all = np.zeros(
        (len(df), raw_dim),
        dtype=np.float32,
    )

    raw_all[valid_mask] = raw_valid

    train_raw = raw_all[train_mask]

    # ---------------------------------------------------------
    # Explained variance used for Table S2
    # ---------------------------------------------------------

    variance = calculate_explained_variance(
        train_raw
    )

    # ---------------------------------------------------------
    # Manuscript representation: fixed 64-D PCA
    # ---------------------------------------------------------

    pca = PCA(
        n_components=EMBEDDING_DIM,
        random_state=RANDOM_STATE,
    )

    pca.fit(train_raw)

    reduced_all = np.zeros(
        (len(df), EMBEDDING_DIM),
        dtype=np.float32,
    )

    reduced_all[valid_mask] = (
        pca
        .transform(raw_all[valid_mask])
        .astype(np.float32)
    )

    # ---------------------------------------------------------
    # Export
    # ---------------------------------------------------------

    channel_dir = (
        output_dir
        / f"channel_{channel_id}"
    )
    channel_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pca_path = (
        channel_dir
        / "pca_64d.joblib"
    )

    joblib.dump(
        pca,
        pca_path,
    )

    # Raw SciBERT vectors are saved as an array for internal
    # visualization/reproducibility but are not required as public
    # source data.
    raw_path = (
        channel_dir
        / "scibert_embeddings.npy"
    )

    np.save(
        raw_path,
        raw_all,
    )

    output_df = df[
        ALIGN_KEYS
        + [
            "split",
            "channel",
            "channel_name",
            "mechanism_text",
            "canonical_tags",
            "evidence_status",
            "embedding_valid",
        ]
    ].copy()

    embedding_columns = [
        f"emb_{i}"
        for i in range(EMBEDDING_DIM)
    ]

    embedding_df = pd.DataFrame(
        reduced_all,
        columns=embedding_columns,
        index=output_df.index,
    )

    output_df = pd.concat(
        [
            output_df,
            embedding_df,
        ],
        axis=1,
    )

    csv_path = (
        channel_dir
        / "reference_embeddings_64d.csv"
    )

    output_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "channel": channel_id,
        "channel_name": channel_name,
        "total_records": int(len(df)),
        "reported_records": int(valid_mask.sum()),
        "reported_training_records": int(
            train_mask.sum()
        ),
        "raw_scibert_dimension": int(raw_dim),
        "retained_dimension": EMBEDDING_DIM,
        "pca_fit_split": "train",
        "pca_explained_variance": {
            str(dim): variance[dim]
            for dim in PCA_REPORT_DIMS
        },
    }

    with (
        channel_dir
        / "embedding_summary.json"
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

    print(
        f"Saved: {csv_path}"
    )
    print(
        f"Saved: {pca_path}"
    )

    return summary


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the six 64-D literature-derived "
            "mechanism reference spaces."
        )
    )

    parser.add_argument(
        "--mechanism-json",
        required=True,
        type=Path,
        help=(
            "Joint six-channel mechanism JSON."
        ),
    )

    parser.add_argument(
        "--split-csv",
        required=True,
        type=Path,
        help=(
            "Fixed train/val/test split manifest."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/mechanism_embeddings"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_df = load_split_manifest(
        args.split_csv
    )

    device = resolve_device(
        args.device
    )

    print(
        f"Loading fixed SciBERT model: "
        f"{SCIBERT_MODEL}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            SCIBERT_MODEL
        )
    )

    model = (
        AutoModel
        .from_pretrained(
            SCIBERT_MODEL
        )
        .to(device)
    )

    summaries: List[Dict[str, Any]] = []

    for channel_id in range(1, 7):
        summary = (
            build_channel_reference_space(
                channel_id=channel_id,
                mechanism_json=args.mechanism_json,
                split_df=split_df,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
            )
        )

        summaries.append(
            summary
        )

    # ---------------------------------------------------------
    # Reproduce Table S2
    # ---------------------------------------------------------

    variance_rows = []

    for summary in summaries:
        row = {
            "Mechanism channel":
                summary["channel_name"]
        }

        for dim in PCA_REPORT_DIMS:
            row[f"{dim}D"] = (
                summary[
                    "pca_explained_variance"
                ][str(dim)]
            )

        variance_rows.append(row)

    variance_df = pd.DataFrame(
        variance_rows
    )

    average_row = {
        "Mechanism channel": "Average"
    }

    for dim in PCA_REPORT_DIMS:
        average_row[f"{dim}D"] = (
            variance_df[f"{dim}D"]
            .mean()
        )

    variance_df = pd.concat(
        [
            variance_df,
            pd.DataFrame(
                [average_row]
            ),
        ],
        ignore_index=True,
    )

    variance_df.to_csv(
        args.output_dir
        / "pca_explained_variance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        args.output_dir
        / "embedding_build_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summaries,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print(
        "Finished building all six "
        "mechanism reference spaces."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()