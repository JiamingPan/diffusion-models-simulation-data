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

from .probe_transforms import Transform


PARAM_NAMES = (
    "Omega_m",
    "sigma_8",
    "A_SN1",
    "A_AGN1",
    "A_SN2",
    "A_AGN2",
)
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


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
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
        "transforms": _json_value(transforms),
        "seeds": _json_value(seeds),
        "arguments": _json_value(arguments),
    }
    if head_path is not None:
        manifest["head"] = _artifact_record(head_path)
    if extra:
        manifest.update(_json_value(extra))
    return manifest
