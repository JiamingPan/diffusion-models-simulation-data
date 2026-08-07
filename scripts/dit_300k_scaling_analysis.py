#!/usr/bin/env python
"""Validated helpers for the focused DiT 300k scaling notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

FRESH_SWEEP_NAME = "nf_generalize_fig2_dit_l16_fresh300k_v2"
FRESH_SAMPLE_LABEL = "dpm50_fresh300k_v2"
FRESH_TRAINING_SEED = 123
FRESH_SCHEDULER = "DPMSolverMultistepScheduler"
FRESH_SAMPLER_STEPS = 50
FRESH_SAMPLE_COUNT = 512


def expected_dataset_tags() -> tuple[str, ...]:
    """Return the ordered dataset tags required by every full-sweep figure."""

    return DATASET_TAGS


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
