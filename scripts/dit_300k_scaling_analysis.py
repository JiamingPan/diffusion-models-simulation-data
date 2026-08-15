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


def flatten_numeric(values: Any) -> np.ndarray:
    """Flatten nested metric payloads while ignoring non-numeric entries."""

    output: list[float] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            for key in ("loss", "value", "mean", "avg"):
                if key in value:
                    visit(value[key])
                    return
            return
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                visit(item)
            return
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            return

    visit(values)
    return np.asarray(output, dtype=float)


def prepare_loss_history(
    metrics: Mapping[str, Any],
    *,
    steps_per_epoch: int,
    target_updates: int,
    restart_updates: int = 4_000,
    minimum_fraction: float = 0.98,
) -> dict[str, Any]:
    """Validate and cycle-average one fresh training-loss history."""

    epoch_loss = flatten_numeric(metrics.get("epoch_loss"))
    if len(epoch_loss) == 0 or not np.all(np.isfinite(epoch_loss)):
        raise ValueError("loss history has no finite epoch_loss values")
    steps_per_epoch = max(1, int(steps_per_epoch))
    target_updates = int(target_updates)

    recorded_updates = len(epoch_loss) * steps_per_epoch
    for key in ("optimizer_updates", "optimizer_step", "global_step", "num_updates", "updates"):
        values = flatten_numeric(metrics.get(key))
        if len(values):
            recorded_updates = int(round(values[-1]))
            break
    if recorded_updates < minimum_fraction * target_updates:
        raise ValueError(
            f"recorded {recorded_updates} optimizer updates; expected at least "
            f"{minimum_fraction:.0%} of {target_updates}"
        )

    epoch_updates = (np.arange(len(epoch_loss), dtype=float) + 1.0) * steps_per_epoch
    window = max(1, min(len(epoch_loss), int(round(restart_updates / steps_per_epoch))))
    if window == 1:
        averaged = epoch_loss.copy()
        averaged_updates = epoch_updates
    else:
        averaged = np.convolve(epoch_loss, np.ones(window) / window, mode="valid")
        averaged_updates = np.convolve(epoch_updates, np.ones(window) / window, mode="valid")

    tail_count = max(1, int(np.ceil(0.05 * len(epoch_loss))))
    return {
        "epoch_loss": epoch_loss,
        "epoch_updates": epoch_updates,
        "updates": averaged_updates,
        "cycle_averaged_loss": averaged,
        "recorded_updates": recorded_updates,
        "epochs_completed": int(len(epoch_loss)),
        "steps_per_epoch": steps_per_epoch,
        "tail_median_loss": float(np.median(epoch_loss[-tail_count:])),
        "tail_q25_loss": float(np.quantile(epoch_loss[-tail_count:], 0.25)),
        "tail_q75_loss": float(np.quantile(epoch_loss[-tail_count:], 0.75)),
        "best_loss": float(np.min(epoch_loss)),
    }


def checkpoint_metric_candidates(checkpoint_path: str | Path) -> list[Path]:
    """Return metrics tied to one exact checkpoint, excluding later run metrics."""

    checkpoint = Path(checkpoint_path)
    match = re.fullmatch(r"checkpoint-epoch-(\d+)", checkpoint.name)
    if match is None:
        raise ValueError(f"invalid checkpoint directory name: {checkpoint}")
    target_epoch = int(match.group(1))

    candidates: list[Path] = []
    if checkpoint.is_dir():
        candidates.extend(sorted(checkpoint.glob("metrics*.json")))
    if checkpoint.parent.is_dir():
        for path in sorted(checkpoint.parent.glob("metrics_epoch_*.json")):
            epoch_match = re.fullmatch(r"metrics_epoch_(\d+)\.json", path.name)
            if epoch_match and int(epoch_match.group(1)) == target_epoch:
                candidates.append(path)
    return list(dict.fromkeys(candidates))


def prepare_stitched_loss_history(
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    *,
    steps_per_epoch: int,
    restart_updates: int = 4_000,
    minimum_fraction: float = 0.98,
) -> dict[str, Any]:
    """Validate and stitch stage-local loss logs onto one global update axis."""

    if not segments:
        raise ValueError("loss history has no continuation segments")
    steps_per_epoch = max(1, int(steps_per_epoch))
    all_losses: list[np.ndarray] = []
    all_updates: list[np.ndarray] = []
    stage_recorded_updates: list[int] = []
    first_start: int | None = None
    previous_end: int | None = None

    for index, (metrics, raw_start, raw_end) in enumerate(segments, start=1):
        start_updates = int(raw_start)
        end_updates = int(raw_end)
        if end_updates <= start_updates:
            raise ValueError(
                f"segment {index} has invalid update interval "
                f"[{start_updates}, {end_updates}]"
            )
        if previous_end is not None and start_updates != previous_end:
            raise ValueError(
                f"segment {index} starts at {start_updates}, expected {previous_end}"
            )

        epoch_loss = flatten_numeric(metrics.get("epoch_loss"))
        if len(epoch_loss) == 0 or not np.all(np.isfinite(epoch_loss)):
            raise ValueError(f"segment {index} has no finite epoch_loss values")
        recorded = int(len(epoch_loss) * steps_per_epoch)
        expected = end_updates - start_updates
        if recorded < minimum_fraction * expected:
            raise ValueError(
                f"segment {index} recorded {recorded} optimizer updates; expected at least "
                f"{minimum_fraction:.0%} of {expected}"
            )
        if recorded > expected / minimum_fraction:
            raise ValueError(
                f"segment {index} recorded {recorded} optimizer updates; expected about "
                f"{expected}, so the metrics are not stage-local"
            )

        epoch_updates = start_updates + (
            np.arange(len(epoch_loss), dtype=float) + 1.0
        ) * steps_per_epoch
        all_losses.append(epoch_loss)
        all_updates.append(epoch_updates)
        stage_recorded_updates.append(recorded)
        first_start = start_updates if first_start is None else first_start
        previous_end = end_updates

    epoch_loss = np.concatenate(all_losses)
    epoch_updates = np.concatenate(all_updates)
    window = max(1, min(len(epoch_loss), int(round(restart_updates / steps_per_epoch))))
    if window == 1:
        averaged = epoch_loss.copy()
        averaged_updates = epoch_updates.copy()
    else:
        averaged = np.convolve(epoch_loss, np.ones(window) / window, mode="valid")
        averaged_updates = np.convolve(epoch_updates, np.ones(window) / window, mode="valid")

    tail_count = max(1, int(np.ceil(0.05 * len(epoch_loss))))
    return {
        "epoch_loss": epoch_loss,
        "epoch_updates": epoch_updates,
        "updates": averaged_updates,
        "cycle_averaged_loss": averaged,
        "start_updates": int(first_start),
        "recorded_updates": int(previous_end),
        "stage_recorded_updates": np.asarray(stage_recorded_updates, dtype=int),
        "epochs_completed": int(len(epoch_loss)),
        "steps_per_epoch": steps_per_epoch,
        "tail_median_loss": float(np.median(epoch_loss[-tail_count:])),
        "tail_q25_loss": float(np.quantile(epoch_loss[-tail_count:], 0.25)),
        "tail_q75_loss": float(np.quantile(epoch_loss[-tail_count:], 0.75)),
        "best_loss": float(np.min(epoch_loss)),
    }


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


def evenly_spaced_indices(*, total: int, count: int) -> np.ndarray:
    """Choose deterministic, unique sample indices spanning an archive."""

    total = int(total)
    count = int(count)
    if total <= 0:
        raise ValueError("total must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if count > total:
        raise ValueError("count cannot exceed total")
    if count == 1:
        return np.asarray([0], dtype=int)
    return np.rint(np.linspace(0, total - 1, count)).astype(int)


def streaming_nearest_neighbors(
    generated: np.ndarray,
    training_batches: Iterable[np.ndarray],
) -> dict[str, Any]:
    """Find each query's MSE-nearest training image without loading all training data.

    Pairwise distances use the squared-norm identity, so memory scales with the
    current training batch rather than query x training x pixels.
    """

    queries = np.asarray(generated, dtype=np.float32)
    if queries.ndim < 2 or len(queries) == 0:
        raise ValueError("generated must contain at least one image")
    image_shape = queries.shape[1:]
    query_flat = queries.reshape(len(queries), -1).astype(np.float64, copy=False)
    pixels = query_flat.shape[1]
    query_norm2 = np.einsum("ij,ij->i", query_flat, query_flat)
    query_norm = np.sqrt(query_norm2)

    best_mse = np.full(len(queries), np.inf, dtype=float)
    best_cosine = np.full(len(queries), np.nan, dtype=float)
    best_index = np.full(len(queries), -1, dtype=int)
    best_images = np.empty((len(queries), *image_shape), dtype=np.float32)
    training_offset = 0

    for batch in training_batches:
        images = np.asarray(batch, dtype=np.float32)
        if images.ndim != queries.ndim or images.shape[1:] != image_shape:
            raise ValueError(
                f"training image shape {images.shape[1:]} does not match "
                f"generated image shape {image_shape}"
            )
        if len(images) == 0:
            continue

        training_flat = images.reshape(len(images), -1).astype(np.float64, copy=False)
        training_norm2 = np.einsum("ij,ij->i", training_flat, training_flat)
        dot = query_flat @ training_flat.T
        pair_mse = (
            query_norm2[:, None] + training_norm2[None, :] - 2.0 * dot
        ) / pixels
        pair_mse = np.maximum(pair_mse, 0.0)

        local_index = np.argmin(pair_mse, axis=1)
        local_mse = pair_mse[np.arange(len(queries)), local_index]
        improved = local_mse < best_mse
        if np.any(improved):
            query_index = np.flatnonzero(improved)
            selected_local = local_index[improved]
            denominator = query_norm[improved] * np.sqrt(training_norm2[selected_local])
            cosine = np.divide(
                dot[query_index, selected_local],
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0,
            )
            best_mse[improved] = local_mse[improved]
            best_cosine[improved] = cosine
            best_index[improved] = training_offset + selected_local
            best_images[improved] = images[selected_local]
        training_offset += len(images)

    if training_offset == 0:
        raise ValueError("no training samples were provided")
    if np.any(best_index < 0):
        raise RuntimeError("nearest-neighbor search did not resolve every query")

    return {
        "nearest_index": best_index,
        "mse": best_mse,
        "cosine_similarity": best_cosine,
        "nearest_images": best_images,
        "n_training": int(training_offset),
    }


def _as_nchw_float(images: np.ndarray) -> np.ndarray:
    """Normalize an image batch to finite ``(N, C, H, W)`` float64 data."""

    batch = np.asarray(images, dtype=np.float64)
    if batch.ndim == 3:
        batch = batch[:, None, :, :]
    if batch.ndim != 4 or len(batch) == 0:
        raise ValueError("images must be a non-empty (N,H,W) or (N,C,H,W) batch")
    if not np.all(np.isfinite(batch)):
        raise ValueError("images contain non-finite values")
    return batch


def _radial_power_geometry(
    image_shape: tuple[int, int],
    *,
    nbins: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return common radial-bin centers and masks for a 2D Fourier grid."""

    height, width = (int(image_shape[0]), int(image_shape[1]))
    nbins = int(nbins)
    if height <= 1 or width <= 1 or nbins <= 0:
        raise ValueError("image dimensions and nbins must be positive")

    ky = np.fft.fftfreq(height) * height
    kx = np.fft.fftfreq(width) * width
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    positive = radius > 0
    edges = np.linspace(radius[positive].min(), radius[positive].max(), nbins + 1)
    masks: list[np.ndarray] = []
    for index in range(nbins):
        mask = positive & (radius >= edges[index]) & (radius < edges[index + 1])
        masks.append(mask)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, tuple(masks)


def _batch_radial_power(
    images: np.ndarray,
    *,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one azimuthally averaged power spectrum per image."""

    batch = _as_nchw_float(images)
    if batch.shape[1] != 1:
        raise ValueError(f"expected one image channel, found {batch.shape[1]}")
    height, width = batch.shape[-2:]
    kbins, masks = _radial_power_geometry((height, width), nbins=nbins)

    centered = batch - batch.mean(axis=(-2, -1), keepdims=True)
    fourier = np.fft.fft2(centered, axes=(-2, -1))
    power = np.abs(fourier) ** 2 / float(height * width)
    power = power[:, 0]

    spectra = np.full((len(batch), len(masks)), np.nan, dtype=np.float64)
    for index, mask in enumerate(masks):
        if np.any(mask):
            spectra[:, index] = power[:, mask].mean(axis=1)
    return kbins, spectra


def _validated_hist_edges(hist_edges: Sequence[float]) -> np.ndarray:
    edges = np.asarray(hist_edges, dtype=np.float64)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("hist_edges must be finite, increasing, and one-dimensional")
    return edges


def aggregate_physical_batches(
    batches: Iterable[np.ndarray],
    *,
    hist_edges: Sequence[float],
    nbins: int,
) -> dict[str, Any]:
    """Stream an exact image subset into a pixel PDF and mean radial spectrum.

    Histogram counts and power sums are accumulated before normalization, so
    the answer is invariant to how the exact subset is partitioned into batches.
    """

    edges = _validated_hist_edges(hist_edges)
    expected_shape: tuple[int, ...] | None = None
    histogram_counts = np.zeros(len(edges) - 1, dtype=np.int64)
    power_sum = np.zeros(int(nbins), dtype=np.float64)
    power_count = np.zeros(int(nbins), dtype=np.int64)
    kbins: np.ndarray | None = None
    n_images = 0
    n_pixels = 0
    in_range_pixels = 0

    for values in batches:
        batch = _as_nchw_float(values)
        if expected_shape is None:
            expected_shape = batch.shape[1:]
        elif batch.shape[1:] != expected_shape:
            raise ValueError(
                f"image shape changed from {expected_shape} to {batch.shape[1:]}"
            )

        batch_counts, _ = np.histogram(batch.reshape(-1), bins=edges)
        histogram_counts += batch_counts
        in_range_pixels += int(batch_counts.sum())
        n_images += int(len(batch))
        n_pixels += int(batch.size)

        batch_kbins, spectra = _batch_radial_power(batch, nbins=int(nbins))
        if kbins is None:
            kbins = batch_kbins
        elif not np.allclose(kbins, batch_kbins, rtol=0.0, atol=0.0):
            raise ValueError("Fourier-bin geometry changed between batches")
        finite = np.isfinite(spectra)
        power_sum += np.nansum(spectra, axis=0)
        power_count += finite.sum(axis=0)

    if n_images == 0 or kbins is None:
        raise ValueError("no images were provided")
    if in_range_pixels == 0:
        raise ValueError("no pixels fall inside hist_edges")

    widths = np.diff(edges)
    histogram = histogram_counts / (float(in_range_pixels) * widths)
    mean_pk = np.divide(
        power_sum,
        power_count,
        out=np.full_like(power_sum, np.nan),
        where=power_count > 0,
    )
    return {
        "hist": histogram,
        "hist_counts": histogram_counts,
        "hist_edges": edges,
        "kbins": kbins,
        "mean_pk": mean_pk,
        "n_images": n_images,
        "n_pixels": n_pixels,
        "pixel_coverage": in_range_pixels / float(n_pixels),
    }


def per_sample_physical_errors(
    images: np.ndarray,
    *,
    reference_hist: Sequence[float],
    hist_edges: Sequence[float],
    reference_mean_pk: Sequence[float],
    nbins: int,
) -> dict[str, np.ndarray]:
    """Retain per-sample PDF and power errors instead of only their means."""

    batch = _as_nchw_float(images)
    edges = _validated_hist_edges(hist_edges)
    reference_histogram = np.asarray(reference_hist, dtype=np.float64)
    if reference_histogram.shape != (len(edges) - 1,):
        raise ValueError(
            "reference histogram length must equal len(hist_edges) - 1"
        )
    if not np.all(np.isfinite(reference_histogram)):
        raise ValueError("reference histogram contains non-finite values")

    reference_pk = np.asarray(reference_mean_pk, dtype=np.float64)
    if reference_pk.shape != (int(nbins),):
        raise ValueError("reference mean power length must equal nbins")

    widths = np.diff(edges)
    sample_histograms = np.empty((len(batch), len(edges) - 1), dtype=np.float64)
    for index, image in enumerate(batch):
        counts, _ = np.histogram(image.reshape(-1), bins=edges)
        included = int(counts.sum())
        if included == 0:
            sample_histograms[index] = np.nan
        else:
            sample_histograms[index] = counts / (float(included) * widths)

    hist_l1 = np.nansum(
        np.abs(sample_histograms - reference_histogram[None, :]) * widths[None, :],
        axis=1,
    )
    kbins, sample_pk = _batch_radial_power(batch, nbins=int(nbins))
    valid_reference = np.isfinite(reference_pk) & (reference_pk > 0)
    pk_ratio = np.divide(
        sample_pk,
        reference_pk[None, :],
        out=np.full_like(sample_pk, np.nan),
        where=valid_reference[None, :],
    )
    valid_ratio = np.isfinite(pk_ratio) & (pk_ratio > 0)
    log_error = np.full_like(pk_ratio, np.nan)
    log_error[valid_ratio] = np.abs(np.log10(pk_ratio[valid_ratio]))
    with np.errstate(invalid="ignore"):
        pk_log10_mae = np.nanmean(log_error, axis=1)

    return {
        "hist_l1": hist_l1,
        "pk_log10_mae": pk_log10_mae,
        "pk_ratio": pk_ratio,
        "per_sample_hist": sample_histograms,
        "per_sample_pk": sample_pk,
        "kbins": kbins,
    }
