"""Shared evaluation and reporting helpers for frozen-probe controls."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import batch_power_spectra
from .probe_transforms import (
    Transform,
    compose_transforms,
    dihedral_transform,
    get_transform,
    roll_transform,
)


PARAM_NAMES = (
    "Omega_m",
    "sigma_8",
    "A_SN1",
    "A_AGN1",
    "A_SN2",
    "A_AGN2",
)
C4_LIMITATION = (
    "The transfer controls match only the measured two-point power deficit. "
    "They do not match the one-point PDF or higher-order structure, so a "
    "negative result rules out only that two-point deficit as the explanation."
)
DEFAULT_K_CUTS = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 52.0, 64.0)
DESCRIPTOR_COLUMNS = (
    "transform",
    "transform_family",
    "k_cut",
    "k_cut_over_knyq",
    "window",
    "dihedral_g",
    "roll_dx",
    "roll_dy",
)


@dataclass(frozen=True)
class TransformSpec:
    name: str
    family: str
    transform: Transform
    k_cut: float | None = None
    window: str | None = None
    dihedral_g: int | None = None
    roll_dx: int | None = None
    roll_dy: int | None = None

    def manifest_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "k_cut": self.k_cut,
            "window": self.window,
            "dihedral_g": self.dihedral_g,
            "roll_dx": self.roll_dx,
            "roll_dy": self.roll_dy,
        }


def build_c0_specs(seed: int) -> tuple[list[TransformSpec], list[tuple[int, int]]]:
    """Enumerate eight dihedral elements crossed with five roll states."""
    rng = np.random.default_rng(int(seed))
    offsets: list[tuple[int, int]] = []
    while len(offsets) < 4:
        dx, dy = (int(value) for value in rng.integers(-63, 64, size=2))
        offset = (dx, dy)
        if offset != (0, 0) and offset not in offsets:
            offsets.append(offset)

    specs: list[TransformSpec] = []
    for element in range(8):
        for dx, dy in [(0, 0), *offsets]:
            if dx == 0 and dy == 0:
                name = "identity" if element == 0 else f"dihedral_g{element}"
                family = "identity" if element == 0 else "dihedral"
                transform = dihedral_transform(element)
            else:
                name = f"dihedral_g{element}__roll_dx{dx}_dy{dy}"
                family = "roll"
                transform = compose_transforms(
                    dihedral_transform(element),
                    roll_transform(dx, dy),
                )
            specs.append(
                TransformSpec(
                    name=name,
                    family=family,
                    transform=transform,
                    dihedral_g=element,
                    roll_dx=dx,
                    roll_dy=dy,
                )
            )
    return specs, offsets


def build_c1_specs(k_cuts: tuple[float, ...] | list[float]) -> list[TransformSpec]:
    """Build identity, FFT null, and complementary C1 scale-cut arms."""
    cutoffs = tuple(float(value) for value in k_cuts)
    if not cutoffs:
        raise ValueError("At least one C1 cutoff is required")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("C1 cutoffs must be unique")
    if any(not np.isfinite(value) or value <= 0.0 or value > 64.0 for value in cutoffs):
        raise ValueError("C1 cutoffs must be finite and lie in (0, 64]")

    specs = [
        TransformSpec("identity", "identity", get_transform("identity")),
        TransformSpec(
            "fft_roundtrip_null",
            "fft_roundtrip_null",
            get_transform("fft_roundtrip_null"),
            k_cut=64.0,
            window="allpass",
        ),
    ]
    for k_cut in cutoffs:
        label = f"{k_cut:g}"
        for family in ("lowpass", "highpass"):
            for window in ("sharp", "hann"):
                name = f"{family}_kcut{label}_{window}"
                specs.append(
                    TransformSpec(
                        name=name,
                        family=family,
                        transform=get_transform(name),
                        k_cut=k_cut,
                        window=window,
                    )
                )
    return specs


def evaluate_transform_specs(
    images: np.ndarray,
    theta_raw: np.ndarray,
    sim_index: np.ndarray,
    z_index: np.ndarray,
    encoder: Any,
    specs: list[TransformSpec],
    *,
    batch_size: int,
    param_names: tuple[str, ...] = PARAM_NAMES,
) -> pd.DataFrame:
    """Apply transforms and emit one tidy prediction row per parameter."""
    array = np.asarray(images, dtype=np.float32)
    theta = np.asarray(theta_raw, dtype=np.float32)
    sims = np.asarray(sim_index, dtype=np.int64).reshape(-1)
    slices = np.asarray(z_index, dtype=np.int64).reshape(-1)
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(f"Expected images shaped (N,1,H,W), got {array.shape}")
    if theta.shape != (len(array), len(param_names)):
        raise ValueError(
            f"theta_raw has shape {theta.shape}; expected {(len(array), len(param_names))}"
        )
    if sims.shape != (len(array),) or slices.shape != (len(array),):
        raise ValueError("sim_index and z_index must contain one value per image")
    if not specs:
        raise ValueError("At least one transform specification is required")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Transform names must be unique")
    if names.count("identity") != 1:
        raise ValueError("Every control run must contain identity exactly once")

    rows: list[dict[str, Any]] = []
    k_nyquist = array.shape[-1] / 2.0
    for spec in specs:
        transformed, diagnostics = spec.transform(array)
        prediction_norm = encoder.predict_norm(
            transformed,
            batch_size=int(batch_size),
        )
        prediction = np.asarray(encoder.norm_to_raw(prediction_norm), dtype=np.float32)
        if prediction.shape != theta.shape:
            raise ValueError(
                f"Encoder returned shape {prediction.shape}; expected {theta.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"Encoder returned non-finite predictions for {spec.name}")
        out_of_range = float(diagnostics["out_of_range_fraction"])
        for image_index in range(len(array)):
            for parameter_index, parameter in enumerate(param_names):
                rows.append(
                    {
                        "transform": spec.name,
                        "transform_family": spec.family,
                        "k_cut": spec.k_cut,
                        "k_cut_over_knyq": (
                            None if spec.k_cut is None else float(spec.k_cut) / k_nyquist
                        ),
                        "window": spec.window,
                        "dihedral_g": spec.dihedral_g,
                        "roll_dx": spec.roll_dx,
                        "roll_dy": spec.roll_dy,
                        "sim_index": int(sims[image_index]),
                        "z_index": int(slices[image_index]),
                        "parameter": parameter,
                        "theta_true": float(theta[image_index, parameter_index]),
                        "theta_pred": float(prediction[image_index, parameter_index]),
                        "out_of_range_fraction": out_of_range,
                    }
                )
    return pd.DataFrame(rows)


def _fit_slope(theta_true: np.ndarray, theta_pred: np.ndarray) -> float:
    if len(theta_true) < 2 or float(np.var(theta_true)) <= 1.0e-30:
        return float("nan")
    return float(np.polyfit(theta_true.astype(float), theta_pred.astype(float), 1)[0])


def _metric_values(theta_true: np.ndarray, theta_pred: np.ndarray) -> dict[str, float]:
    error = theta_pred - theta_true
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "slope": _fit_slope(theta_true, theta_pred),
    }


def _bootstrap_metric_intervals(
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    values: dict[str, list[float]] = {"rmse": [], "bias": [], "slope": []}
    for _ in range(int(n_boot)):
        indices = rng.integers(0, len(theta_true), size=len(theta_true))
        metrics = _metric_values(theta_true[indices], theta_pred[indices])
        for name, value in metrics.items():
            if np.isfinite(value):
                values[name].append(value)

    intervals: dict[str, float] = {}
    for name, samples in values.items():
        if samples:
            low, high = np.quantile(samples, [0.16, 0.84])
            intervals[f"{name}_ci_low"] = float(low)
            intervals[f"{name}_ci_high"] = float(high)
        else:
            intervals[f"{name}_ci_low"] = float("nan")
            intervals[f"{name}_ci_high"] = float("nan")
    return intervals


def _group_record(columns: tuple[str, ...], key: Any) -> dict[str, Any]:
    values = key if isinstance(key, tuple) else (key,)
    record: dict[str, Any] = {}
    for column, value in zip(columns, values):
        if pd.isna(value):
            record[column] = None
        elif isinstance(value, np.generic):
            record[column] = value.item()
        else:
            record[column] = value
    return record


def aggregate_prediction_table(
    predictions: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Report RMSE, bias, and slope at slice and cosmology grains."""
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")
    required = set(DESCRIPTOR_COLUMNS) | {
        "sim_index",
        "z_index",
        "parameter",
        "theta_true",
        "theta_pred",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    per_slice = predictions.copy()
    cosmology_groups = list(DESCRIPTOR_COLUMNS) + ["sim_index", "parameter"]
    per_cosmology = (
        predictions.groupby(cosmology_groups, dropna=False, as_index=False)
        .agg(theta_true=("theta_true", "first"), theta_pred=("theta_pred", "median"))
    )

    rng = np.random.default_rng(int(seed))
    metric_rows: list[dict[str, Any]] = []
    group_columns = DESCRIPTOR_COLUMNS + ("parameter",)
    for grain, table in (("per_slice", per_slice), ("per_cosmology", per_cosmology)):
        for key, group in table.groupby(list(group_columns), dropna=False, sort=False):
            truth = group["theta_true"].to_numpy(dtype=np.float64)
            prediction = group["theta_pred"].to_numpy(dtype=np.float64)
            record = _group_record(group_columns, key)
            record.update(
                {
                    "grain": grain,
                    "n": int(len(group)),
                    **_metric_values(truth, prediction),
                    **_bootstrap_metric_intervals(
                        truth,
                        prediction,
                        n_boot=int(n_boot),
                        rng=rng,
                    ),
                }
            )
            metric_rows.append(record)
    return {
        "metrics": metric_rows,
        "bootstrap": {"n_resamples": int(n_boot), "seed": int(seed)},
    }


def _spread_table(
    table: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return (
        table.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            transform_std=("theta_pred", lambda values: float(np.std(values, ddof=0))),
            transform_range=(
                "theta_pred",
                lambda values: float(np.max(values) - np.min(values)),
            ),
            n_views=("theta_pred", "size"),
        )
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _bootstrap_median_interval(
    values: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    medians = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        sample = rng.integers(0, len(finite), size=len(finite))
        medians[index] = np.median(finite[sample])
    low, high = np.quantile(medians, [0.16, 0.84])
    return float(low), float(high)


def c0_symmetry_summary(
    predictions: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Compare C0 view spreads with the identity within-simulation baseline."""
    omega = predictions[predictions["parameter"] == "Omega_m"].copy()
    if omega.empty:
        raise ValueError("C0 predictions contain no Omega_m rows")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")

    c0 = omega[
        omega[["dihedral_g", "roll_dx", "roll_dy"]].notna().all(axis=1)
    ].copy()
    if c0.empty:
        raise ValueError("Prediction table contains no C0 descriptor rows")

    identity = c0[c0["transform"] == "identity"]
    baseline = (
        identity.groupby("sim_index", as_index=False)
        .agg(
            within_sim_std=(
                "theta_pred",
                lambda values: float(np.std(values, ddof=0)),
            ),
            within_sim_range=(
                "theta_pred",
                lambda values: float(np.max(values) - np.min(values)),
            ),
            n_slices=("z_index", "nunique"),
        )
    )
    baseline_lookup = baseline.set_index("sim_index").to_dict("index")

    no_roll = c0[(c0["roll_dx"] == 0) & (c0["roll_dy"] == 0)]
    dihedral = _spread_table(no_roll, ["sim_index", "z_index"])
    if not (dihedral["n_views"] == 8).all():
        raise ValueError("Each C0 dihedral slice must contain eight no-roll views")
    dihedral["family"] = "dihedral"

    roll_by_element = _spread_table(
        c0,
        ["sim_index", "z_index", "dihedral_g"],
    )
    if not (roll_by_element["n_views"] == 5).all():
        raise ValueError("Each C0 roll group must contain no-roll plus four rolls")
    roll = (
        roll_by_element.groupby(["sim_index", "z_index"], as_index=False)
        .agg(
            transform_std=("transform_std", "median"),
            transform_range=("transform_range", "median"),
            n_views=("n_views", "sum"),
        )
    )
    roll["family"] = "roll"

    per_slice: list[dict[str, Any]] = []
    for table in (dihedral, roll):
        for row in table.to_dict("records"):
            sim_index = int(row["sim_index"])
            baseline_row = baseline_lookup[sim_index]
            per_slice.append(
                {
                    "family": row["family"],
                    "sim_index": sim_index,
                    "z_index": int(row["z_index"]),
                    "transform_std": float(row["transform_std"]),
                    "transform_range": float(row["transform_range"]),
                    "within_sim_std": float(baseline_row["within_sim_std"]),
                    "within_sim_range": float(baseline_row["within_sim_range"]),
                    "std_over_within_sim_std": _safe_ratio(
                        float(row["transform_std"]),
                        float(baseline_row["within_sim_std"]),
                    ),
                    "range_over_within_sim_range": _safe_ratio(
                        float(row["transform_range"]),
                        float(baseline_row["within_sim_range"]),
                    ),
                }
            )

    rng = np.random.default_rng(int(seed))
    per_slice_table = pd.DataFrame(per_slice)
    family_summary: dict[str, Any] = {}
    for family, group in per_slice_table.groupby("family", sort=False):
        per_sim = (
            group.groupby("sim_index", as_index=False)
            .agg(
                std_ratio=("std_over_within_sim_std", "median"),
                range_ratio=("range_over_within_sim_range", "median"),
            )
        )
        std_values = per_sim["std_ratio"].to_numpy(dtype=np.float64)
        range_values = per_sim["range_ratio"].to_numpy(dtype=np.float64)
        std_low, std_high = _bootstrap_median_interval(
            std_values,
            n_boot=int(n_boot),
            rng=rng,
        )
        range_low, range_high = _bootstrap_median_interval(
            range_values,
            n_boot=int(n_boot),
            rng=rng,
        )
        family_summary[str(family)] = {
            "n_simulations": int(len(per_sim)),
            "median_std_ratio": float(np.nanmedian(std_values)),
            "median_std_ratio_ci_low": std_low,
            "median_std_ratio_ci_high": std_high,
            "median_range_ratio": float(np.nanmedian(range_values)),
            "median_range_ratio_ci_low": range_low,
            "median_range_ratio_ci_high": range_high,
        }

    return {
        "baseline": {
            "definition": "identity Omega_m spread across z-slices within each simulation",
            "per_simulation": baseline.to_dict("records"),
        },
        "per_slice": per_slice,
        "family_summary": family_summary,
        "bootstrap": {"n_resamples": int(n_boot), "seed": int(seed)},
    }


def c1_scale_cut_summary(
    predictions: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Return C1 RMSE, bias, and slope curves at both report grains."""
    allowed = {"identity", "fft_roundtrip_null", "lowpass", "highpass"}
    table = predictions[predictions["transform_family"].isin(allowed)].copy()
    if table.empty:
        raise ValueError("Prediction table contains no C1 transforms")
    report = aggregate_prediction_table(table, n_boot=int(n_boot), seed=int(seed))
    out_of_range = (
        table.groupby("transform", as_index=True)["out_of_range_fraction"]
        .mean()
        .to_dict()
    )
    curves = report["metrics"]
    for row in curves:
        row["out_of_range_fraction"] = float(out_of_range[row["transform"]])
    return {
        "curves": curves,
        "bootstrap": report["bootstrap"],
        "filter_space": "log/tanh-normalized probe input before the VGG [-1,1] clamp",
        "interpretation": (
            "Use low-pass/high-pass asymmetry together with RMSE, signed bias, "
            "slope, the FFT round-trip null, and out-of-range fraction."
        ),
    }


def deterministic_cosmology_split(
    sim_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split unique cosmologies into deterministic equal derivation/evaluation halves."""
    indices = np.asarray(sim_indices, dtype=np.int64).reshape(-1)
    if len(indices) < 2 or len(indices) % 2:
        raise ValueError("sim_indices must contain an even number of cosmologies")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("sim_indices must be unique")
    permutation = np.random.default_rng(int(seed)).permutation(indices)
    midpoint = len(permutation) // 2
    return np.sort(permutation[:midpoint]), np.sort(permutation[midpoint:])


def fit_gaussian_smoothing(k_bins: np.ndarray, power_ratio: np.ndarray) -> float:
    """Fit log R(k) = -sigma^2 k^2 through the physical zero intercept."""
    k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    ratio = np.asarray(power_ratio, dtype=np.float64).reshape(-1)
    if k.shape != ratio.shape or len(k) == 0:
        raise ValueError("k_bins and power_ratio must be non-empty equal-length vectors")
    valid = np.isfinite(k) & np.isfinite(ratio) & (k > 0.0) & (ratio > 0.0)
    if not valid.any():
        raise ValueError("No finite positive power-ratio bins are available")
    k2 = k[valid] ** 2
    log_ratio = np.log(np.clip(ratio[valid], 1.0e-30, None))
    denominator = float(np.dot(k2, k2))
    if denominator <= 0.0:
        raise ValueError("Gaussian fit has zero k leverage")
    sigma_squared = max(0.0, -float(np.dot(k2, log_ratio)) / denominator)
    return float(np.sqrt(sigma_squared))


def power_ratio_transfer(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    *,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure R(k)=P_generated/P_real and return its amplitude transfer."""
    real = np.asarray(real_images, dtype=np.float32)
    generated = np.asarray(generated_images, dtype=np.float32)
    if real.ndim != 4 or generated.ndim != 4:
        raise ValueError("real_images and generated_images must have shape (N,C,H,W)")
    if real.shape[1:] != generated.shape[1:]:
        raise ValueError("Real and generated image shapes must match")
    if len(real) == 0 or len(generated) == 0:
        raise ValueError("Real and generated image sets must be non-empty")
    real_power, k_bins = batch_power_spectra(real, nbins=int(nbins))
    generated_power, generated_k = batch_power_spectra(generated, nbins=int(nbins))
    if not np.allclose(k_bins, generated_k, equal_nan=True):
        raise ValueError("Real and generated power bins differ")
    real_mean = np.nanmean(real_power, axis=0)
    generated_mean = np.nanmean(generated_power, axis=0)
    ratio = generated_mean / np.clip(real_mean, 1.0e-30, None)
    if not (
        np.isfinite(k_bins).all()
        and np.isfinite(real_mean).all()
        and np.isfinite(generated_mean).all()
        and np.isfinite(ratio).all()
    ):
        raise ValueError("Power-ratio transfer contains non-finite bins")
    transfer = np.sqrt(np.clip(ratio, 0.0, None))
    return k_bins, real_mean, generated_mean, ratio, transfer


def subset_generated_cosmologies(
    samples: np.ndarray,
    theta_raw: np.ndarray,
    heldout_indices: np.ndarray,
    *,
    samples_per_cosmology: int,
    selected_simulations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select complete generated blocks while preserving sample/target pairing."""
    array = np.asarray(samples, dtype=np.float32)
    theta = np.asarray(theta_raw, dtype=np.float32)
    heldout = np.asarray(heldout_indices, dtype=np.int64).reshape(-1)
    selected = set(
        np.asarray(selected_simulations, dtype=np.int64).reshape(-1).astype(int).tolist()
    )
    count = int(samples_per_cosmology)
    if count < 1:
        raise ValueError("samples_per_cosmology must be positive")
    if len(array) != len(heldout) * count:
        raise ValueError("Generated samples do not match heldout block ordering")
    if theta.ndim != 2 or theta.shape[0] != len(heldout):
        raise ValueError("theta_raw must contain one parameter row per heldout cosmology")
    if not selected.issubset(set(heldout.astype(int).tolist())):
        raise ValueError("selected_simulations contains an unknown cosmology")

    sample_blocks: list[np.ndarray] = []
    theta_blocks: list[np.ndarray] = []
    sim_rows: list[int] = []
    sample_rows: list[int] = []
    for heldout_offset, sim_index in enumerate(heldout):
        if int(sim_index) not in selected:
            continue
        start = heldout_offset * count
        stop = start + count
        sample_blocks.append(array[start:stop])
        theta_blocks.append(np.repeat(theta[heldout_offset : heldout_offset + 1], count, axis=0))
        sim_rows.extend([int(sim_index)] * count)
        sample_rows.extend(range(count))
    if not sample_blocks:
        raise ValueError("selected_simulations retained no generated samples")
    return (
        np.concatenate(sample_blocks, axis=0),
        np.concatenate(theta_blocks, axis=0),
        np.asarray(sim_rows, dtype=np.int64),
        np.asarray(sample_rows, dtype=np.int64),
    )


def file_sha256(path: str | Path, block_size: int = 1 << 20) -> str:
    artifact = Path(path).resolve()
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_sklearn_version() -> str:
    return importlib.metadata.version("scikit-learn")


def git_state(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"revision": revision, "dirty": bool(status.strip())}


def _artifact_record(path: str | Path) -> dict[str, str]:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return {"path": str(artifact), "sha256": file_sha256(artifact)}


def json_safe(value: Any) -> Any:
    """Recursively convert a report payload to strict JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def build_run_manifest(
    *,
    project_dir: str | Path,
    encoder_path: str | Path,
    head_path: str | Path | None,
    heldout_indices: np.ndarray,
    slices_per_sim: int,
    transforms: list[dict[str, Any]],
    seeds: dict[str, int],
    arguments: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build complete provenance for one frozen-probe control invocation."""
    heldout = np.asarray(heldout_indices, dtype=np.int64).reshape(-1)
    if heldout.size == 0:
        raise ValueError("heldout_indices is empty")
    names = [str(record.get("name")) for record in transforms]
    if names.count("identity") != 1:
        raise ValueError("Manifest transforms must contain identity exactly once")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(project_dir),
        "encoder": _artifact_record(encoder_path),
        "scikit_learn_version": installed_sklearn_version(),
        "heldout_indices": heldout.astype(int).tolist(),
        "slices_per_sim": int(slices_per_sim),
        "transforms": json_safe(transforms),
        "seeds": json_safe(seeds),
        "arguments": json_safe(arguments),
    }
    if head_path is not None:
        manifest["head"] = _artifact_record(head_path)
    if extra:
        manifest.update(json_safe(extra))
    return manifest
