"""Physics-aware and reproducibility metrics for generated 2D fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def radial_power_spectrum_2d(field: np.ndarray, nbins: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Compute an isotropic 2D power spectrum for one image.

    The field mean is subtracted before the FFT.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D field, got shape {arr.shape}.")

    arr = arr - arr.mean()
    fft = np.fft.fftn(arr)
    power = (fft * fft.conj()).real / arr.size

    ky = np.fft.fftfreq(arr.shape[0]) * arr.shape[0]
    kx = np.fft.fftfreq(arr.shape[1]) * arr.shape[1]
    kkx, kky = np.meshgrid(kx, ky)
    kvals = np.sqrt(kkx**2 + kky**2)

    valid = kvals > 0
    edges = np.linspace(kvals[valid].min(), kvals[valid].max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    pk = np.full(nbins, np.nan, dtype=np.float64)
    for i in range(nbins):
        mask = (kvals >= edges[i]) & (kvals < edges[i + 1])
        if mask.any():
            pk[i] = power[mask].mean()
    return pk, centers


def batch_power_spectra(images: np.ndarray, nbins: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial 2D power spectra for ``(N, C, H, W)`` images."""
    arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f"Expected images shaped (N,C,H,W), got {arr.shape}.")

    pks = []
    kbins = None
    for image in arr:
        pk, kbins = radial_power_spectrum_2d(image[0], nbins=nbins)
        pks.append(pk)
    return np.asarray(pks), np.asarray(kbins)


def power_spectrum_summary(real: np.ndarray, generated: np.ndarray, nbins: int = 25) -> dict[str, float]:
    """Summarize generated/real power-spectrum mismatch."""
    pk_real, _ = batch_power_spectra(real, nbins=nbins)
    pk_gen, _ = batch_power_spectra(generated, nbins=nbins)

    real_mean = np.nanmean(pk_real, axis=0)
    gen_mean = np.nanmean(pk_gen, axis=0)
    ratio = gen_mean / np.clip(real_mean, 1e-30, None)
    log_abs = np.abs(np.log10(np.clip(ratio, 1e-30, None)))

    finite = np.where(np.isfinite(ratio))[0]
    thirds = np.array_split(finite, 3) if len(finite) else [[], [], []]
    band_means = [
        float(np.nanmean(ratio[idx])) if len(idx) else float("nan")
        for idx in thirds
    ]

    return {
        "pk_log10_mae": float(np.nanmean(log_abs)),
        "pk_ratio_low_k": band_means[0],
        "pk_ratio_mid_k": band_means[1],
        "pk_ratio_high_k": band_means[2],
    }


def field_histogram(images: np.ndarray, bins: int = 120, value_range: tuple[float, float] = (-1, 1)) -> dict[str, object]:
    """Return histogram and one-point statistics for flattened field values."""
    vals = np.asarray(images, dtype=np.float64).ravel()
    hist, edges = np.histogram(vals, bins=bins, range=value_range, density=True)
    quantiles = np.quantile(vals, [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0])
    return {
        "hist": hist.tolist(),
        "bin_edges": edges.tolist(),
        "min": float(quantiles[0]),
        "q001": float(quantiles[1]),
        "q01": float(quantiles[2]),
        "q10": float(quantiles[3]),
        "median": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "q999": float(quantiles[7]),
        "max": float(quantiles[8]),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "frac_abs_ge_0999": float(np.mean(np.abs(vals) >= 0.999)),
    }


def nearest_neighbor_distances(
    real: np.ndarray,
    generated: np.ndarray,
    max_real: int | None = None,
    max_generated: int | None = None,
) -> dict[str, float]:
    """Compute generated-to-real nearest-neighbor distances in pixel space.

    This is a simple memorization diagnostic. It flattens images and reports
    each generated image's Euclidean distance to its closest real image.
    """
    real_flat = np.asarray(real, dtype=np.float64).reshape(len(real), -1)
    gen_flat = np.asarray(generated, dtype=np.float64).reshape(len(generated), -1)

    if max_real is not None:
        real_flat = real_flat[:max_real]
    if max_generated is not None:
        gen_flat = gen_flat[:max_generated]

    dmins = []
    for g in gen_flat:
        diff = real_flat - g[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        dmins.append(np.sqrt(d2.min()))
    dmins_arr = np.asarray(dmins)

    return {
        "nn_mean": float(dmins_arr.mean()),
        "nn_median": float(np.median(dmins_arr)),
        "nn_min": float(dmins_arr.min()),
        "nn_max": float(dmins_arr.max()),
    }


def nearest_training_matches(
    generated: np.ndarray,
    training: np.ndarray,
    *,
    max_generated: int | None = 8,
    max_training: int | None = None,
    training_chunk: int = 256,
) -> dict[str, np.ndarray]:
    """Match generated images to their nearest training images in pixel MSE.

    The training set is scanned in chunks so the comparison does not allocate
    the full generated-by-training distance matrix.
    """
    generated_arr = np.asarray(generated, dtype=np.float32)
    training_arr = np.asarray(training, dtype=np.float32)
    if generated_arr.ndim == 3:
        generated_arr = generated_arr[:, None]
    if training_arr.ndim == 3:
        training_arr = training_arr[:, None]
    if generated_arr.ndim != 4 or training_arr.ndim != 4:
        raise ValueError(
            "Expected generated and training arrays shaped (N,C,H,W) or (N,H,W)."
        )
    if generated_arr.shape[1:] != training_arr.shape[1:]:
        raise ValueError(
            f"Generated shape {generated_arr.shape[1:]} does not match "
            f"training shape {training_arr.shape[1:]}."
        )
    if len(generated_arr) == 0:
        raise ValueError("generated sample set is empty")
    if len(training_arr) == 0:
        raise ValueError("training reference is empty")
    if int(training_chunk) < 1:
        raise ValueError("training_chunk must be >= 1")

    n_generated = len(generated_arr) if max_generated is None else min(len(generated_arr), int(max_generated))
    n_training = len(training_arr) if max_training is None else min(len(training_arr), int(max_training))
    if n_generated < 1:
        raise ValueError("max_generated must retain at least one sample")
    if n_training < 1:
        raise ValueError("max_training must retain at least one training image")

    generated_flat = generated_arr[:n_generated].reshape(n_generated, -1)
    training_flat = training_arr[:n_training].reshape(n_training, -1)
    generated_norm = np.linalg.norm(generated_flat, axis=1)
    training_norm = np.linalg.norm(training_flat, axis=1)

    nearest_index = np.full(n_generated, -1, dtype=np.int64)
    nearest_mse = np.full(n_generated, np.inf, dtype=np.float64)
    nearest_cosine = np.full(n_generated, np.nan, dtype=np.float64)
    for generated_index, generated_image in enumerate(generated_flat):
        for start in range(0, n_training, int(training_chunk)):
            chunk = training_flat[start : start + int(training_chunk)]
            difference = chunk - generated_image[None]
            mse = np.mean(difference * difference, axis=1)
            local_index = int(np.argmin(mse))
            if float(mse[local_index]) < nearest_mse[generated_index]:
                nearest_mse[generated_index] = float(mse[local_index])
                nearest_index[generated_index] = start + local_index

        match_index = int(nearest_index[generated_index])
        denominator = float(generated_norm[generated_index] * training_norm[match_index])
        nearest_cosine[generated_index] = (
            float(np.dot(generated_image, training_flat[match_index]) / denominator)
            if denominator > 0
            else 0.0
        )

    return {
        "generated_index": np.arange(n_generated, dtype=np.int64),
        "nearest_training_index": nearest_index,
        "nearest_mse": nearest_mse,
        "nearest_rmse": np.sqrt(nearest_mse),
        "nearest_cosine": nearest_cosine,
    }


def _feature_matrix(features: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features), got {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _positive_semidefinite_sqrt(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def frechet_feature_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Compute a stable Fréchet distance between two feature distributions.

    This is the Gaussian two-Wasserstein distance used by FID, but it accepts
    any domain-relevant feature representation rather than requiring ImageNet
    Inception features.
    """
    first_matrix = _feature_matrix(first, "first")
    second_matrix = _feature_matrix(second, "second")
    if first_matrix.shape[1] != second_matrix.shape[1]:
        raise ValueError(
            "first and second feature dimensions must match, got "
            f"{first_matrix.shape[1]} and {second_matrix.shape[1]}"
        )

    first_mean = first_matrix.mean(axis=0)
    second_mean = second_matrix.mean(axis=0)
    first_covariance = np.atleast_2d(np.cov(first_matrix, rowvar=False))
    second_covariance = np.atleast_2d(np.cov(second_matrix, rowvar=False))

    first_sqrt = _positive_semidefinite_sqrt(first_covariance)
    covariance_product = first_sqrt @ second_covariance @ first_sqrt
    covariance_product_sqrt = _positive_semidefinite_sqrt(covariance_product)

    mean_term = float(np.dot(first_mean - second_mean, first_mean - second_mean))
    covariance_term = float(
        np.trace(first_covariance)
        + np.trace(second_covariance)
        - 2.0 * np.trace(covariance_product_sqrt)
    )
    return max(0.0, mean_term + covariance_term)


def real_split_frechet_baseline(features: np.ndarray, seed: int = 0) -> dict[str, float | int]:
    """Estimate finite-sample Fréchet noise by comparing two real-data halves."""
    matrix = _feature_matrix(features, "features")
    half = len(matrix) // 2
    if half < 2:
        raise ValueError("features must contain at least four samples for a split baseline")

    indices = np.random.default_rng(seed).permutation(len(matrix))[: 2 * half]
    first = matrix[indices[:half]]
    second = matrix[indices[half:]]
    return {
        "distance": frechet_feature_distance(first, second),
        "n_first": int(len(first)),
        "n_second": int(len(second)),
        "seed": int(seed),
    }


@dataclass
class ReproducibilityPair:
    """Pairwise comparison between two generated sample sets."""

    first: str
    second: str
    mean_abs_mean_diff: float
    std_abs_diff: float
    pk_log10_mae_between_sets: float


def reproducibility_summary(sample_sets: dict[str, np.ndarray], nbins: int = 25) -> list[dict[str, float | str]]:
    """Compare multiple generated sets to each other.

    This checks whether different random seeds/checkpoints produce statistically
    similar sample distributions.
    """
    names = list(sample_sets)
    out: list[dict[str, float | str]] = []
    pk_means = {}
    for name, samples in sample_sets.items():
        pk, _ = batch_power_spectra(samples, nbins=nbins)
        pk_means[name] = np.nanmean(pk, axis=0)

    for i, name_i in enumerate(names):
        vals_i = np.asarray(sample_sets[name_i]).ravel()
        for name_j in names[i + 1:]:
            vals_j = np.asarray(sample_sets[name_j]).ravel()
            ratio = pk_means[name_i] / np.clip(pk_means[name_j], 1e-30, None)
            out.append({
                "first": name_i,
                "second": name_j,
                "mean_abs_mean_diff": float(abs(vals_i.mean() - vals_j.mean())),
                "std_abs_diff": float(abs(vals_i.std() - vals_j.std())),
                "pk_log10_mae_between_sets": float(np.nanmean(np.abs(np.log10(np.clip(ratio, 1e-30, None))))),
            })
    return out
