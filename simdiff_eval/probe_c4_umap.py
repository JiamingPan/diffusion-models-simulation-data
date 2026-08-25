"""Auditable feature-space metrics for the frozen-probe C4 UMAP analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


IDENTITY_RTOL = 1e-6
IDENTITY_ATOL = 1e-7

PROVENANCE_COLUMNS = [
    "run_name",
    "dataset_size",
    "sim_index",
    "sample_index",
    "slice_index",
    "transform",
    "source",
    "Omega_m",
    "code_revision",
    "config_sha256",
]


def measured_power_deficit_depth(power_ratio: Any) -> float:
    """Return max(0, 1 - min ratio) over the already-saved finite bins."""
    values = np.asarray(power_ratio, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("saved power ratio contains no finite bins")
    return float(max(0.0, 1.0 - float(finite.min())))


def interval_has_zero_width(low: float, high: float) -> bool:
    """Treat only machine-scale numerical width as zero."""
    low = float(low)
    high = float(high)
    if not np.isfinite([low, high]).all():
        return False
    tolerance = 8.0 * np.finfo(np.float64).eps * max(1.0, abs(low), abs(high))
    return bool(abs(high - low) <= tolerance)


def classify_transform_identity(
    original: np.ndarray,
    transformed: np.ndarray,
    metric: dict[str, Any],
    *,
    rtol: float = IDENTITY_RTOL,
    atol: float = IDENTITY_ATOL,
) -> dict[str, bool | str]:
    """Classify a transformed-real arm without changing any scientific metric."""
    left = np.asarray(original, dtype=np.float32)
    right = np.asarray(transformed, dtype=np.float32)
    arrays_allclose = bool(
        left.shape == right.shape
        and np.allclose(left, right, rtol=float(rtol), atol=float(atol))
    )
    centroid_zero = interval_has_zero_width(
        metric["centroid_distance_ci_low"], metric["centroid_distance_ci_high"]
    )
    knn_zero = interval_has_zero_width(
        metric["knn_cross_source_fraction_ci_low"],
        metric["knn_cross_source_fraction_ci_high"],
    )
    is_identity = arrays_allclose and centroid_zero and knn_zero
    if is_identity:
        reason = "transform had no effect at this N"
    else:
        failures = []
        if not arrays_allclose:
            failures.append("transformed arrays differ from original")
        if not centroid_zero:
            failures.append("centroid bootstrap interval has nonzero width")
        if not knn_zero:
            failures.append("kNN bootstrap interval has nonzero width")
        reason = "; ".join(failures)
    return {
        "transform_arrays_allclose": arrays_allclose,
        "centroid_ci_zero_width": centroid_zero,
        "knn_ci_zero_width": knn_zero,
        "transform_is_identity": is_identity,
        "identity_reason": reason,
    }


def generated_identity_diagnostics() -> dict[str, bool | str]:
    """Return explicit not-applicable identity fields for generated samples."""
    return {
        "transform_arrays_allclose": False,
        "centroid_ci_zero_width": False,
        "knn_ci_zero_width": False,
        "transform_is_identity": False,
        "identity_reason": "not applicable: generated samples are not a transform arm",
    }


def perfect_mixing_expectation(source_count: int, reference_count: int) -> float:
    """Exact cross-label-neighbour probability with self excluded."""
    source_count = int(source_count)
    reference_count = int(reference_count)
    if source_count <= 0 or reference_count <= 0:
        raise ValueError("source and reference counts must both be positive")
    total = source_count + reference_count
    return float(
        2.0
        * source_count
        * reference_count
        / (total * (total - 1))
    )


def deterministic_balanced_real_split(
    sim_index: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split original-real rows equally inside each simulation block."""
    simulations = np.asarray(sim_index, dtype=np.int64).reshape(-1)
    if simulations.size == 0:
        raise ValueError("simulation labels are empty")
    rng = np.random.default_rng(int(seed))
    left_blocks = []
    right_blocks = []
    membership: dict[str, dict[str, list[int]]] = {}
    for sim in np.unique(simulations):
        indices = np.flatnonzero(simulations == sim)
        if len(indices) < 2 or len(indices) % 2:
            raise ValueError(
                "every simulation block must contain an even number of at least two rows"
            )
        shuffled = rng.permutation(indices)
        half = len(indices) // 2
        left = np.sort(shuffled[:half])
        right = np.sort(shuffled[half:])
        left_blocks.append(left)
        right_blocks.append(right)
        membership[str(int(sim))] = {
            "left_indices": left.astype(int).tolist(),
            "right_indices": right.astype(int).tolist(),
        }
    left_indices = np.concatenate(left_blocks)
    right_indices = np.concatenate(right_blocks)
    return left_indices, right_indices, {
        "seed": int(seed),
        "rule": "seeded within-simulation balanced half split",
        "by_simulation": membership,
    }


def real_split_mixing_baseline(
    features: np.ndarray,
    sim_index: np.ndarray,
    *,
    k: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Apply the production mixing statistic to two deterministic real halves."""
    values = np.asarray(features)
    simulations = np.asarray(sim_index, dtype=np.int64).reshape(-1)
    if len(values) != len(simulations):
        raise ValueError("features and simulation labels are misaligned")
    left, right, membership = deterministic_balanced_real_split(
        simulations, seed=int(seed)
    )
    metric = compare_source_to_reference(
        values[left],
        values[right],
        source_sim=simulations[left],
        reference_sim=simulations[right],
        k=int(k),
        n_boot=int(n_boot),
        seed=int(seed),
    )
    return {
        "real_split_mixing_baseline": metric["knn_cross_source_fraction"],
        "real_split_mixing_baseline_ci_low": metric[
            "knn_cross_source_fraction_ci_low"
        ],
        "real_split_mixing_baseline_ci_high": metric[
            "knn_cross_source_fraction_ci_high"
        ],
        "split_membership": membership,
    }


def frozen_mlp_inputs(head: Any, raw_features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply only the already-fitted scaler that feeds the frozen MLP dense layer."""
    features = np.asarray(raw_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"raw_features must be a nonempty matrix; got {features.shape}")
    named_steps = getattr(head, "named_steps", None)
    if named_steps is None or not hasattr(named_steps, "get"):
        raise TypeError("Frozen head must be a fitted sklearn Pipeline")
    if list(named_steps)[:2] != ["standardscaler", "mlpregressor"]:
        raise TypeError(
            "Frozen head pipeline order must be standardscaler then mlpregressor"
        )
    scaler = named_steps.get("standardscaler")
    mlp = named_steps.get("mlpregressor")
    if scaler is None or mlp is None:
        raise TypeError(
            "Frozen head must contain standardscaler followed by mlpregressor"
        )
    expected = int(getattr(scaler, "n_features_in_", -1))
    if expected != features.shape[1]:
        raise ValueError(
            f"Frozen scaler expects {expected} features; got {features.shape[1]}"
        )
    transformed = np.asarray(scaler.transform(features), dtype=np.float32)
    if transformed.shape != features.shape or not np.isfinite(transformed).all():
        raise ValueError("Frozen scaler returned invalid MLP inputs")
    return transformed, {
        "feature_dim": int(features.shape[1]),
        "scaler_class": type(scaler).__name__,
        "head_class": type(mlp).__name__,
        "operation": "transform_only",
    }


def balanced_real_slice_pairs(
    heldout_indices: np.ndarray, *, slices_per_sim: int = 64
) -> np.ndarray:
    """Choose a fixed, parameter-independent, evenly spaced slice set."""
    heldout = np.asarray(heldout_indices, dtype=np.int64).reshape(-1)
    count = int(slices_per_sim)
    if heldout.size == 0:
        raise ValueError("heldout_indices is empty")
    if count < 1 or count > 128 or 128 % count != 0:
        raise ValueError("slices_per_sim must be a positive divisor of 128")
    slice_indices = np.arange(0, 128, 128 // count, dtype=np.int64)
    if len(slice_indices) != count:
        raise AssertionError("fixed slice construction produced the wrong count")
    return np.column_stack(
        [np.repeat(heldout, count), np.tile(slice_indices, len(heldout))]
    )


def _centroid_distance(source: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(source, dtype=np.float64).mean(axis=0)
            - np.asarray(reference, dtype=np.float64).mean(axis=0)
        )
    )


def _knn_cross_source_by_sim(
    source: np.ndarray,
    reference: np.ndarray,
    source_sim: np.ndarray,
    reference_sim: np.ndarray,
    *,
    k: int,
) -> dict[int, float]:
    from sklearn.neighbors import NearestNeighbors

    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    labels = np.concatenate(
        [np.ones(len(source), dtype=np.int8), np.zeros(len(reference), dtype=np.int8)]
    )
    sims = np.concatenate(
        [np.asarray(source_sim, dtype=np.int64), np.asarray(reference_sim, dtype=np.int64)]
    )
    combined = np.concatenate([source, reference], axis=0)
    neighbors = int(k)
    if neighbors < 1 or neighbors >= len(combined):
        raise ValueError("k must be positive and smaller than the combined sample count")
    candidates = NearestNeighbors(n_neighbors=neighbors + 1).fit(combined).kneighbors(
        combined, return_distance=False
    )
    indices = np.empty((len(combined), neighbors), dtype=np.int64)
    for row_index, row in enumerate(candidates):
        nonself = row[row != row_index]
        if len(nonself) < neighbors:
            raise RuntimeError("nearest-neighbour query did not return enough nonself rows")
        indices[row_index] = nonself[:neighbors]
    cross_fraction = (labels[indices] != labels[:, None]).mean(axis=1)
    return {
        int(sim): float(cross_fraction[sims == sim].mean())
        for sim in np.unique(sims)
    }


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def compare_source_to_reference(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    source_sim: np.ndarray,
    reference_sim: np.ndarray,
    k: int = 15,
    n_boot: int = 2000,
    seed: int = 123,
) -> dict[str, float | int]:
    """Compare balanced clusters and bootstrap the 32 simulation blocks."""
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    source_sim = np.asarray(source_sim, dtype=np.int64).reshape(-1)
    reference_sim = np.asarray(reference_sim, dtype=np.int64).reshape(-1)
    if source.ndim != 2 or reference.ndim != 2 or source.shape[1] != reference.shape[1]:
        raise ValueError("source and reference must be feature matrices of equal width")
    if len(source) != len(source_sim) or len(reference) != len(reference_sim):
        raise ValueError("feature rows and simulation labels are misaligned")
    simulations = np.intersect1d(np.unique(source_sim), np.unique(reference_sim))
    if len(simulations) < 2:
        raise ValueError("at least two shared simulation blocks are required")
    if set(np.unique(source_sim)) != set(simulations) or set(np.unique(reference_sim)) != set(simulations):
        raise ValueError("source and reference must contain the same simulations")
    source_counts = {int(sim): int(np.sum(source_sim == sim)) for sim in simulations}
    reference_counts = {int(sim): int(np.sum(reference_sim == sim)) for sim in simulations}
    if source_counts != reference_counts:
        raise ValueError("source and reference must be balanced within every simulation")

    centroid = _centroid_distance(source, reference)
    mixing_by_sim = _knn_cross_source_by_sim(
        source, reference, source_sim, reference_sim, k=int(k)
    )
    mixing = float(np.mean([mixing_by_sim[int(sim)] for sim in simulations]))

    rng = np.random.default_rng(int(seed))
    centroid_boot = np.empty(int(n_boot), dtype=np.float64)
    mixing_boot = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        sampled = rng.choice(simulations, size=len(simulations), replace=True)
        source_rows = np.concatenate([np.flatnonzero(source_sim == sim) for sim in sampled])
        reference_rows = np.concatenate(
            [np.flatnonzero(reference_sim == sim) for sim in sampled]
        )
        centroid_boot[index] = _centroid_distance(
            source[source_rows], reference[reference_rows]
        )
        mixing_boot[index] = np.mean([mixing_by_sim[int(sim)] for sim in sampled])
    centroid_low, centroid_high = _percentile_ci(centroid_boot)
    mixing_low, mixing_high = _percentile_ci(mixing_boot)
    return {
        "n_simulations": int(len(simulations)),
        "samples_per_source": int(len(source)),
        "k": int(k),
        "centroid_distance": centroid,
        "centroid_distance_ci_low": centroid_low,
        "centroid_distance_ci_high": centroid_high,
        "knn_cross_source_fraction": mixing,
        "knn_cross_source_fraction_ci_low": mixing_low,
        "knn_cross_source_fraction_ci_high": mixing_high,
    }


def source_metadata(
    *,
    run_name: str,
    dataset_size: int,
    source: str,
    transform: str,
    sim_index: np.ndarray,
    sample_index: np.ndarray,
    slice_index: np.ndarray,
    omega_m: np.ndarray,
    code_revision: str,
    config_sha256: str,
) -> pd.DataFrame:
    """Build one provenance row per feature vector."""
    arrays = [
        np.asarray(sim_index).reshape(-1),
        np.asarray(sample_index).reshape(-1),
        np.asarray(slice_index).reshape(-1),
        np.asarray(omega_m).reshape(-1),
    ]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError("sample provenance arrays are misaligned")
    count = lengths.pop()
    return pd.DataFrame(
        {
            "run_name": [str(run_name)] * count,
            "dataset_size": [int(dataset_size)] * count,
            "sim_index": arrays[0].astype(np.int64),
            "sample_index": arrays[1].astype(np.int64),
            "slice_index": arrays[2].astype(np.int64),
            "transform": [str(transform)] * count,
            "source": [str(source)] * count,
            "Omega_m": arrays[3].astype(float),
            "code_revision": [str(code_revision)] * count,
            "config_sha256": [str(config_sha256)] * count,
        },
        columns=PROVENANCE_COLUMNS,
    )
