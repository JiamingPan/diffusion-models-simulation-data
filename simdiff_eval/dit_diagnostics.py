"""Diagnostics for DiT continuation fidelity and patch-grid artifacts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def bootstrap_mean_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 123,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for the mean."""
    vector = _finite_vector(values, "values")
    confidence = float(confidence)
    n_resamples = int(n_resamples)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    if np.all(vector == vector[0]):
        value = float(vector[0])
        return value, value

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    max_elements = 2_000_000
    chunk_size = max(1, max_elements // vector.size)
    for start in range(0, n_resamples, chunk_size):
        stop = min(n_resamples, start + chunk_size)
        indices = rng.integers(0, vector.size, size=(stop - start, vector.size))
        means[start:stop] = vector[indices].mean(axis=1)

    alpha = 0.5 * (1.0 - confidence)
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _percentile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    alpha = 0.5 * (1.0 - confidence)
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_histogram_l1_interval(
    generated: np.ndarray,
    reference_probability: np.ndarray,
    hist_edges: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 123,
) -> tuple[float, float]:
    """Bootstrap map-level uncertainty in one-point histogram L1 distance."""
    images = np.asarray(generated)
    reference = _finite_vector(reference_probability, "reference_probability")
    edges = _finite_vector(hist_edges, "hist_edges")
    if images.ndim < 2 or images.shape[0] < 1:
        raise ValueError("generated must contain at least one image")
    if not np.isfinite(images).all():
        raise ValueError("generated contains non-finite values")
    if len(edges) != len(reference) + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("hist_edges must define reference_probability bins")
    if not np.isclose(reference.sum(), 1.0):
        raise ValueError("reference_probability must sum to one")
    n_resamples = int(n_resamples)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    counts = np.stack(
        [np.histogram(image, bins=edges)[0] for image in images]
    ).astype(np.float64)
    if counts.sum() < 1:
        raise ValueError("generated images contain no in-range histogram values")
    rng = np.random.default_rng(seed)
    statistics = np.empty(n_resamples, dtype=np.float64)
    max_elements = 2_000_000
    chunk_size = max(1, max_elements // (len(images) * len(reference)))
    for start in range(0, n_resamples, chunk_size):
        stop = min(n_resamples, start + chunk_size)
        indices = rng.integers(0, len(images), size=(stop - start, len(images)))
        resampled_counts = counts[indices].sum(axis=1)
        totals = resampled_counts.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("a bootstrap resample contains no in-range pixels")
        probabilities = resampled_counts / totals[:, None]
        statistics[start:stop] = np.abs(probabilities - reference).sum(axis=1)
    return _percentile_interval(statistics, confidence)


def bootstrap_power_log10_mae_interval(
    generated_power_spectra: np.ndarray,
    real_mean_power: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 123,
) -> tuple[float, float]:
    """Bootstrap map-level uncertainty in mean absolute log10 power error."""
    spectra = np.asarray(generated_power_spectra, dtype=np.float64)
    real_mean = _finite_vector(real_mean_power, "real_mean_power")
    if spectra.ndim != 2 or spectra.shape[0] < 1:
        raise ValueError("generated_power_spectra must have shape (n_samples, n_bins)")
    if spectra.shape[1] != len(real_mean):
        raise ValueError("generated and real power spectra must have the same bins")
    if not np.isfinite(spectra).all() or np.any(real_mean <= 0):
        raise ValueError("power spectra must be finite and real mean power positive")
    n_resamples = int(n_resamples)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(seed)
    statistics = np.empty(n_resamples, dtype=np.float64)
    max_elements = 2_000_000
    chunk_size = max(1, max_elements // spectra.size)
    for start in range(0, n_resamples, chunk_size):
        stop = min(n_resamples, start + chunk_size)
        indices = rng.integers(0, len(spectra), size=(stop - start, len(spectra)))
        means = spectra[indices].mean(axis=1)
        ratios = means / real_mean[None, :]
        statistics[start:stop] = np.abs(
            np.log10(np.clip(ratios, 1.0e-30, None))
        ).mean(axis=1)
    return _percentile_interval(statistics, confidence)


def selected_power_bin_statistics(
    power_spectra: np.ndarray,
    bin_indices: Iterable[int] = (20, 40, 60),
    n_resamples: int = 2000,
    seed: int = 123,
) -> list[dict[str, float | int]]:
    """Summarize sample-to-sample power variation at selected radial bins."""
    spectra = np.asarray(power_spectra, dtype=np.float64)
    if spectra.ndim != 2:
        raise ValueError(
            "power_spectra must be a two-dimensional array shaped "
            "(n_samples, n_bins)"
        )
    if spectra.shape[0] < 1 or spectra.shape[1] < 1:
        raise ValueError("power_spectra must contain samples and k-bins")

    rows: list[dict[str, float | int]] = []
    for offset, raw_index in enumerate(bin_indices):
        index = int(raw_index)
        if index < 0 or index >= spectra.shape[1]:
            raise ValueError(
                f"k-bin {index} is outside the valid range 0..{spectra.shape[1] - 1}"
            )
        values = spectra[:, index]
        if not np.isfinite(values).all():
            raise ValueError(f"k-bin {index} contains non-finite values")
        interval = bootstrap_mean_interval(
            values, n_resamples=n_resamples, seed=int(seed) + offset
        )
        rows.append(
            {
                "k_bin": index,
                "n_samples": int(values.size),
                "mean": float(values.mean()),
                "variance": float(values.var(ddof=1)) if values.size > 1 else 0.0,
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "mean_ci_low": interval[0],
                "mean_ci_high": interval[1],
            }
        )
    return rows


def two_sample_selected_power_ratio_statistics(
    generated_power_spectra: np.ndarray,
    real_selected_power: np.ndarray,
    *,
    bin_indices: Iterable[int] = (20, 40, 60),
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 123,
) -> list[dict[str, float | int]]:
    """Bootstrap selected-k generated/real ratios by resampling both samples."""
    generated = np.asarray(generated_power_spectra, dtype=np.float64)
    real = np.asarray(real_selected_power, dtype=np.float64)
    indices = tuple(int(index) for index in bin_indices)
    if generated.ndim != 2 or generated.shape[0] < 1:
        raise ValueError("generated_power_spectra must have shape (n_samples, n_bins)")
    if real.ndim != 2 or real.shape != (real.shape[0], len(indices)) or real.shape[0] < 1:
        raise ValueError(
            "real_selected_power must have shape (n_real_samples, len(bin_indices))"
        )
    if not np.isfinite(generated).all() or not np.isfinite(real).all():
        raise ValueError("selected power samples must be finite")
    n_resamples = int(n_resamples)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    rows: list[dict[str, float | int]] = []
    for offset, index in enumerate(indices):
        if index < 0 or index >= generated.shape[1]:
            raise ValueError(
                f"k-bin {index} is outside the valid range 0..{generated.shape[1] - 1}"
            )
        generated_values = generated[:, index]
        real_values = real[:, offset]
        real_mean = float(real_values.mean())
        if real_mean <= 0:
            raise ValueError(f"real mean power at k-bin {index} must be positive")

        rng = np.random.default_rng(int(seed) + offset)
        ratios = np.empty(n_resamples, dtype=np.float64)
        max_elements = 2_000_000
        chunk_size = max(
            1,
            max_elements // (len(generated_values) + len(real_values)),
        )
        for start in range(0, n_resamples, chunk_size):
            stop = min(n_resamples, start + chunk_size)
            generated_indices = rng.integers(
                0,
                len(generated_values),
                size=(stop - start, len(generated_values)),
            )
            real_indices = rng.integers(
                0,
                len(real_values),
                size=(stop - start, len(real_values)),
            )
            generated_means = generated_values[generated_indices].mean(axis=1)
            real_means = real_values[real_indices].mean(axis=1)
            if np.any(real_means <= 0):
                raise ValueError(f"bootstrap real mean at k-bin {index} is not positive")
            ratios[start:stop] = generated_means / real_means

        interval = _percentile_interval(ratios, confidence)
        rows.append(
            {
                "k_bin": index,
                "n_generated": int(len(generated_values)),
                "n_real": int(len(real_values)),
                "mean": float(generated_values.mean() / real_mean),
                "mean_ci_low": interval[0],
                "mean_ci_high": interval[1],
                "real_pk_sem": (
                    float(real_values.std(ddof=1) / np.sqrt(len(real_values)))
                    if len(real_values) > 1
                    else 0.0
                ),
            }
        )
    return rows


def one_point_l1_common_bins(
    real: np.ndarray,
    generated: np.ndarray,
    bins: int = 120,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Integrate absolute PDF disagreement using exactly shared histogram bins."""
    real_values = _finite_vector(real, "real")
    generated_values = _finite_vector(generated, "generated")
    bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be positive")
    low, high = map(float, value_range)
    if not low < high:
        raise ValueError("value_range must be strictly increasing")

    edges = np.linspace(low, high, bins + 1, dtype=np.float64)
    real_hist, _ = np.histogram(real_values, bins=edges, density=True)
    generated_hist, _ = np.histogram(generated_values, bins=edges, density=True)
    return float(np.sum(np.abs(real_hist - generated_hist) * np.diff(edges)))


def _axis_patch_series(
    differences: np.ndarray,
    *,
    boundary_mask: np.ndarray,
    control_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    boundary = differences[..., boundary_mask]
    control = differences[..., control_mask]
    if boundary.size == 0 or control.size == 0:
        raise ValueError("patch geometry must contain boundary and control edges")
    if boundary.shape[-1] != control.shape[-1]:
        raise ValueError("patch boundary and control samples must have equal size")
    reduce_axes = tuple(range(1, boundary.ndim))
    return boundary.mean(axis=reduce_axes), control.mean(axis=reduce_axes)


def patch_boundary_per_image(
    images: np.ndarray,
    patch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Return per-image patch-boundary and equal-size control discontinuities."""
    array = np.asarray(images, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"images must have shape (N,C,H,W); found {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("images must contain at least one image and channel")
    if not np.isfinite(array).all():
        raise ValueError("images contain non-finite values")

    patch_size = int(patch_size)
    if patch_size < 2:
        raise ValueError("patch_size must be at least two pixels")
    height, width = array.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image dimensions {(height, width)} must be divisible by patch_size={patch_size}"
        )

    horizontal_differences = np.abs(np.diff(array, axis=-2)).transpose(0, 1, 3, 2)
    vertical_differences = np.abs(np.diff(array, axis=-1))
    horizontal_positions = np.arange(1, height)
    vertical_positions = np.arange(1, width)
    horizontal_boundary_positions = horizontal_positions[
        (horizontal_positions % patch_size) == 0
    ]
    vertical_boundary_positions = vertical_positions[
        (vertical_positions % patch_size) == 0
    ]
    horizontal_boundary, horizontal_control = _axis_patch_series(
        horizontal_differences,
        boundary_mask=np.isin(horizontal_positions, horizontal_boundary_positions),
        control_mask=np.isin(
            horizontal_positions, horizontal_boundary_positions - 1
        ),
    )
    vertical_boundary, vertical_control = _axis_patch_series(
        vertical_differences,
        boundary_mask=np.isin(vertical_positions, vertical_boundary_positions),
        control_mask=np.isin(vertical_positions, vertical_boundary_positions - 1),
    )
    return {
        "boundary": 0.5 * (horizontal_boundary + vertical_boundary),
        "control": 0.5 * (horizontal_control + vertical_control),
        "horizontal_boundary": horizontal_boundary,
        "horizontal_control": horizontal_control,
        "vertical_boundary": vertical_boundary,
        "vertical_control": vertical_control,
    }


def summarize_patch_boundary_series(
    series: dict[str, np.ndarray],
    patch_size: int = 8,
) -> dict[str, float | int]:
    """Summarize per-image boundary jumps against equal-size local controls."""
    required = {
        "boundary",
        "control",
        "horizontal_boundary",
        "horizontal_control",
        "vertical_boundary",
        "vertical_control",
    }
    if set(series) != required:
        raise ValueError(f"patch series keys must be exactly {sorted(required)}")
    values = {key: _finite_vector(value, key) for key, value in series.items()}
    sizes = {value.size for value in values.values()}
    if len(sizes) != 1:
        raise ValueError("patch series must contain the same number of images")

    boundary = values["boundary"]
    control = values["control"]
    boundary_mean = float(boundary.mean())
    control_mean = float(control.mean())
    ratio = boundary_mean / control_mean if control_mean > 0.0 else float("inf")
    boundary_median = float(np.median(boundary))
    control_median = float(np.median(control))
    median_ratio = (
        boundary_median / control_median if control_median > 0.0 else float("inf")
    )
    excess = boundary_mean - control_mean

    def axis_ratio(boundary_key: str, control_key: str) -> float:
        numerator = float(values[boundary_key].mean())
        denominator = float(values[control_key].mean())
        return numerator / denominator if denominator > 0.0 else float("inf")

    return {
        "patch_size": int(patch_size),
        "n_images": int(boundary.size),
        "boundary_mean_abs_difference": boundary_mean,
        "control_mean_abs_difference": control_mean,
        "boundary_median_abs_difference": boundary_median,
        "control_median_abs_difference": control_median,
        "boundary_to_control_ratio": float(ratio),
        "boundary_to_control_median_ratio": float(median_ratio),
        "boundary_excess_abs_difference": float(excess),
        "boundary_relative_excess": (
            float(excess / control_mean) if control_mean > 0.0 else float("inf")
        ),
        "horizontal_boundary_to_control_ratio": float(
            axis_ratio("horizontal_boundary", "horizontal_control")
        ),
        "vertical_boundary_to_control_ratio": float(
            axis_ratio("vertical_boundary", "vertical_control")
        ),
    }


def patch_boundary_statistics(
    images: np.ndarray,
    patch_size: int = 8,
) -> dict[str, float | int]:
    """Compare adjacent-pixel jumps on and away from the DiT patch grid."""
    series = patch_boundary_per_image(images, patch_size=patch_size)
    summary = summarize_patch_boundary_series(series, patch_size=patch_size)
    # Compatibility aliases retain older notebook column names.
    summary["interior_mean_abs_difference"] = summary["control_mean_abs_difference"]
    summary["boundary_to_interior_ratio"] = summary["boundary_to_control_ratio"]
    summary["horizontal_boundary_to_interior_ratio"] = summary[
        "horizontal_boundary_to_control_ratio"
    ]
    summary["vertical_boundary_to_interior_ratio"] = summary[
        "vertical_boundary_to_control_ratio"
    ]
    return summary
