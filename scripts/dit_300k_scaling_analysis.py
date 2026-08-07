#!/usr/bin/env python
"""Validated helpers for the focused DiT 300k scaling notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DATASET_POWERS = tuple(range(6, 16))
DATASET_TAGS = tuple(f"d2p{power:02d}" for power in DATASET_POWERS)
DATASET_SIZES = tuple(2**power for power in DATASET_POWERS)

DIT_UPDATE_BUDGETS = {
    "dit_l8": 200_000,
    "dit_base": 200_000,
    "dit_l16": 300_000,
}
DIT_LABELS = {
    "dit_l8": "DiT-L8 200k",
    "dit_base": "DiT-L12 / base 200k",
    "dit_l16": "DiT-L16 fresh 300k",
}
UNET_LABELS = {
    "u64": "UNet-64 historical 200k",
    "u128": "UNet-128 historical 200k",
    "u256": "UNet-256 historical 200k",
}
MODEL_PARAMETER_COUNTS = {
    "u64": 26_621_057,
    "dit_l8": 95_000_000,
    "dit_base": 138_290_000,
    "u128": 140_539_521,
    "dit_l16": 182_000_000,
    "u256": 196_059_905,
}

FRESH_SWEEP_NAME = "nf_generalize_fig2_dit_l16_fresh300k_v2"
FRESH_SAMPLE_LABEL = "dpm50_fresh300k_v2"
FRESH_TRAINING_SEED = 123
FRESH_SCHEDULER = "DPMSolverMultistepScheduler"
FRESH_SAMPLER_STEPS = 50
FRESH_SAMPLE_COUNT = 512


def expected_dataset_tags() -> tuple[str, ...]:
    """Return the ordered dataset tags required by every full-sweep figure."""

    return DATASET_TAGS


def _dataset_tag_from_text(value: Any) -> str | None:
    match = re.search(r"d2p(\d+)", str(value))
    if match is None:
        return None
    return f"d2p{int(match.group(1)):02d}"


def _dataset_size_from_tag(value: Any) -> float:
    tag = _dataset_tag_from_text(value)
    return float(2 ** int(tag[-2:])) if tag is not None else float("nan")


def normalize_generalization_table(
    table: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Normalize dataset identifiers and generalization-score column names."""

    if table.empty:
        raise ValueError(f"{context}: table is empty")
    result = table.copy()
    if "arch" not in result.columns:
        raise ValueError(f"{context}: missing columns ['arch']")

    if "dataset_tag" not in result.columns:
        source_column = "run_name" if "run_name" in result.columns else None
        if source_column is None:
            raise ValueError(f"{context}: cannot derive dataset_tag without run_name")
        result["dataset_tag"] = result[source_column].map(_dataset_tag_from_text)
    else:
        result["dataset_tag"] = result["dataset_tag"].map(_dataset_tag_from_text)

    if "dataset_size" not in result.columns:
        result["dataset_size"] = result["dataset_tag"].map(_dataset_size_from_tag)
    result["dataset_size"] = pd.to_numeric(result["dataset_size"], errors="coerce")

    for quantile in ("q50", "q68", "q90", "q95", "q99"):
        score_column = f"gen_gl_{quantile}"
        copy_column = f"gen_copy_fraction_{quantile}"
        legacy_copy_column = f"copy_fraction_{quantile}"
        if score_column not in result.columns and copy_column in result.columns:
            result[score_column] = 1.0 - pd.to_numeric(
                result[copy_column], errors="coerce"
            )
        if score_column not in result.columns and legacy_copy_column in result.columns:
            result[score_column] = 1.0 - pd.to_numeric(
                result[legacy_copy_column], errors="coerce"
            )

    invalid_tags = sorted(
        result.loc[result["dataset_tag"].isna()].index.astype(str).tolist()
    )
    if invalid_tags:
        raise ValueError(f"{context}: could not derive dataset tags for rows {invalid_tags}")
    return result


def build_mixed_dit_metric_table(
    historical_table: pd.DataFrame,
    fresh_l16_table: pd.DataFrame,
    *,
    feature: str,
    score_column: str = "gen_gl_q95",
) -> pd.DataFrame:
    """Build the L8/L12 200k plus independently trained L16 300k table."""

    historical = normalize_generalization_table(
        historical_table, context=f"historical {feature} DiT"
    )
    fresh = normalize_generalization_table(
        fresh_l16_table, context=f"fresh independent L16 300k {feature}"
    )

    frames: list[pd.DataFrame] = []
    for arch in ("dit_l8", "dit_base"):
        selected = require_exact_dataset_sweep(
            historical,
            arch=arch,
            value_columns=(score_column,),
            context=f"historical {feature} {DIT_LABELS[arch]}",
        )
        selected["updates_k"] = DIT_UPDATE_BUDGETS[arch] // 1_000
        selected["model_label"] = DIT_LABELS[arch]
        selected["source"] = "historical fixed 200k"
        frames.append(selected)

    selected = require_exact_dataset_sweep(
        fresh,
        arch="dit_l16",
        value_columns=(score_column,),
        context=f"fresh independent L16 300k {feature}",
    )
    selected["updates_k"] = DIT_UPDATE_BUDGETS["dit_l16"] // 1_000
    selected["model_label"] = DIT_LABELS["dit_l16"]
    selected["source"] = "fresh independent 300k v2"
    frames.append(selected)

    result = pd.concat(frames, ignore_index=True)
    result["feature"] = str(feature).upper()
    return result


def build_historical_unet_metric_table(
    table: pd.DataFrame,
    *,
    feature: str,
    score_column: str = "gen_gl_q95",
) -> pd.DataFrame:
    """Build complete historical 200k UNet reference curves."""

    normalized = normalize_generalization_table(
        table, context=f"historical UNet {feature}"
    )
    frames: list[pd.DataFrame] = []
    for arch in ("u64", "u128", "u256"):
        selected = require_exact_dataset_sweep(
            normalized,
            arch=arch,
            value_columns=(score_column,),
            context=f"historical {UNET_LABELS[arch].replace(' 200k', '')}",
        )
        selected["updates_k"] = 200
        selected["model_label"] = UNET_LABELS[arch]
        selected["source"] = "historical fixed 200k"
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    result["feature"] = str(feature).upper()
    return result


def require_exact_dataset_sweep(
    table: pd.DataFrame,
    *,
    arch: str,
    value_columns: Iterable[str],
    context: str,
) -> pd.DataFrame:
    """Select one finite row per required data size for ``arch`` or fail."""

    required_columns = {"arch", "dataset_tag", "dataset_size", *value_columns}
    missing_columns = sorted(required_columns - set(table.columns))
    if missing_columns:
        raise ValueError(f"{context}: missing columns {missing_columns}")

    selected = table[table["arch"].astype(str) == arch].copy()
    tags = selected["dataset_tag"].astype(str)
    observed = set(tags)
    missing_tags = sorted(set(DATASET_TAGS) - observed)
    extra_tags = sorted(observed - set(DATASET_TAGS))
    duplicate_tags = sorted(tags[tags.duplicated(keep=False)].unique().tolist())

    invalid_values: dict[str, list[str]] = {}
    for column in value_columns:
        values = pd.to_numeric(selected[column], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            invalid_values[column] = tags.iloc[np.flatnonzero(bad)].tolist()

    sizes_by_tag = selected.set_index("dataset_tag")["dataset_size"].to_dict()
    wrong_sizes = {
        tag: sizes_by_tag.get(tag)
        for tag, expected in zip(DATASET_TAGS, DATASET_SIZES)
        if tag in sizes_by_tag and int(sizes_by_tag[tag]) != expected
    }

    if (
        len(selected) != len(DATASET_TAGS)
        or missing_tags
        or extra_tags
        or duplicate_tags
        or invalid_values
        or wrong_sizes
    ):
        raise ValueError(
            f"{context}: invalid sweep rows={len(selected)}, "
            f"missing={missing_tags}, extra={extra_tags}, "
            f"duplicates={duplicate_tags}, invalid_values={invalid_values}, "
            f"wrong_sizes={wrong_sizes}"
        )

    order = {tag: index for index, tag in enumerate(DATASET_TAGS)}
    selected["_dataset_order"] = tags.map(order)
    return (
        selected.sort_values("_dataset_order")
        .drop(columns="_dataset_order")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class N50Result:
    n50: float
    status: str
    interval: tuple[float, float] | None
    crossing_count: int


def interpolate_n50(
    dataset_sizes: Sequence[float],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> N50Result:
    """Interpolate a unique upward threshold crossing in log2 data size."""

    sizes = np.asarray(dataset_sizes, dtype=float)
    values = np.asarray(scores, dtype=float)
    if sizes.ndim != 1 or values.ndim != 1 or len(sizes) != len(values):
        raise ValueError("dataset_sizes and scores must be equal-length 1D arrays")
    if len(sizes) == 0 or not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(values)):
        raise ValueError("dataset_sizes and scores must be finite and non-empty")
    if np.any(sizes <= 0):
        raise ValueError("dataset_sizes must be positive")

    order = np.argsort(sizes)
    sizes = sizes[order]
    values = values[order]
    if np.any(np.diff(sizes) <= 0):
        raise ValueError("dataset_sizes must be unique")

    centered = values - threshold
    exact = np.flatnonzero(centered == 0)
    sign_changes = np.flatnonzero(centered[:-1] * centered[1:] < 0)
    crossing_count = int(len(exact) + len(sign_changes))

    if np.all(centered >= 0):
        return N50Result(float(sizes[0]), "left_censored", None, crossing_count)
    if np.all(centered < 0):
        return N50Result(float(sizes[-1]), "right_censored", None, crossing_count)
    if crossing_count != 1:
        return N50Result(float("nan"), "ambiguous", None, crossing_count)

    if len(exact) == 1:
        index = int(exact[0])
        return N50Result(float(sizes[index]), "crossing", (float(sizes[index]), float(sizes[index])), 1)

    index = int(sign_changes[0])
    if centered[index] > 0 or centered[index + 1] < 0:
        return N50Result(float("nan"), "ambiguous", None, crossing_count)

    x0, x1 = np.log2(sizes[index : index + 2])
    y0, y1 = values[index : index + 2]
    fraction = (threshold - y0) / (y1 - y0)
    n50 = float(2 ** (x0 + fraction * (x1 - x0)))
    return N50Result(
        n50,
        "crossing",
        (float(sizes[index]), float(sizes[index + 1])),
        1,
    )


def summarize_n50(
    table: pd.DataFrame,
    *,
    score_column: str = "gen_gl_q95",
) -> pd.DataFrame:
    """Summarize q95 transition locations without hiding censoring."""

    required = {
        "feature",
        "arch",
        "model_label",
        "updates_k",
        "dataset_size",
        score_column,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"transition summary: missing columns {missing}")

    rows: list[dict[str, Any]] = []
    groups = table.groupby(["feature", "arch", "model_label"], sort=True)
    for (feature, arch, label), group in groups:
        update_values = pd.to_numeric(group["updates_k"], errors="coerce").dropna().unique()
        if len(update_values) != 1:
            raise ValueError(
                f"transition summary: {feature}/{arch} has update budgets {update_values.tolist()}"
            )
        result = interpolate_n50(group["dataset_size"], group[score_column])
        rows.append(
            {
                "feature": feature,
                "arch": arch,
                "model_label": label,
                "updates_k": int(update_values[0]),
                "score_column": score_column,
                "n50": result.n50,
                "log2_n50": float(np.log2(result.n50)) if np.isfinite(result.n50) else np.nan,
                "status": result.status,
                "crossing_count": result.crossing_count,
                "interval_low": result.interval[0] if result.interval else np.nan,
                "interval_high": result.interval[1] if result.interval else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["feature", "updates_k", "arch"]).reset_index(drop=True)


def scalar_value(value: Any) -> Any:
    """Convert a scalar NumPy archive field to its Python value."""

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar metadata, found shape {array.shape}")
    return array.reshape(()).item()


def _normalized_path(value: Any) -> str:
    return str(Path(str(scalar_value(value))).expanduser().resolve())


def validate_sample_archive_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_checkpoint: str | Path,
    expected_config_path: str | Path,
    expected_scheduler: str,
    expected_num_steps: int,
    expected_seed: int,
    expected_samples: int,
) -> dict[str, Any]:
    """Validate one generated-sample archive against the frozen run manifest."""

    required = {
        "requested_checkpoint",
        "resolved_checkpoint",
        "config_path",
        "scheduler",
        "num_steps",
        "seed",
        "samples",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"sample archive is missing fields {missing}")

    expected_checkpoint_text = str(Path(expected_checkpoint).expanduser().resolve())
    expected_config_text = str(Path(expected_config_path).expanduser().resolve())
    requested = _normalized_path(metadata["requested_checkpoint"])
    resolved = _normalized_path(metadata["resolved_checkpoint"])
    config_path = _normalized_path(metadata["config_path"])
    scheduler = str(scalar_value(metadata["scheduler"]))
    num_steps = int(scalar_value(metadata["num_steps"]))
    seed = int(scalar_value(metadata["seed"]))
    samples = np.asarray(metadata["samples"])

    checks = (
        (requested == expected_checkpoint_text, "requested checkpoint", requested, expected_checkpoint_text),
        (resolved == expected_checkpoint_text, "resolved checkpoint", resolved, expected_checkpoint_text),
        (config_path == expected_config_text, "config path", config_path, expected_config_text),
        (scheduler == expected_scheduler, "scheduler", scheduler, expected_scheduler),
        (num_steps == expected_num_steps, "step count", num_steps, expected_num_steps),
        (seed == expected_seed, "seed", seed, expected_seed),
        (len(samples) == expected_samples, "sample count", len(samples), expected_samples),
    )
    for ok, label, observed, expected in checks:
        if not ok:
            raise ValueError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")

    return {
        "requested_checkpoint": requested,
        "resolved_checkpoint": resolved,
        "config_path": config_path,
        "scheduler": scheduler,
        "num_steps": num_steps,
        "seed": seed,
        "n_generated": int(len(samples)),
    }
