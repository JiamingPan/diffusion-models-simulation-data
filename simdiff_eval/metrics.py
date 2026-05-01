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
