"""Read-only diagnostics for saved continuously conditioned HI UNet samples.

The comparison is deliberately in the generator's log+tanh model space. It
does not claim that model-space FFT power is the physical HI power spectrum.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def as_maps(images: np.ndarray, name: str) -> np.ndarray:
    """Return finite scalar maps shaped (N,H,W), preserving memory mapping."""
    maps = np.asarray(images)
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    if maps.ndim != 3 or len(maps) == 0:
        raise ValueError(f"{name} must have shape (N,H,W) or (N,1,H,W)")
    return maps


def normalize_raw_hi(images: np.ndarray, kwargs: dict) -> np.ndarray:
    """Match the frozen log+tanh preprocessing used for conditional training."""
    raw = as_maps(images, "raw HI images").astype(np.float32, copy=False)
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("raw HI images must be finite and nonnegative")
    center = np.float32(kwargs["center"])
    xmax = np.float32(kwargs["xmax"])
    if not np.isfinite(center) or not np.isfinite(xmax) or xmax <= 0:
        raise ValueError("invalid frozen image normalization")
    alpha = np.float32(kwargs.get("alpha", 0.8))
    beta = np.float32(kwargs.get("beta", 10.0))
    gamma = np.float32(kwargs.get("gamma", 1.0))
    delta = np.float32(kwargs.get("delta", 1.0))
    sigma = np.float32(kwargs.get("sigma", 1.5))
    x = (np.log(np.maximum(raw, np.float32(1e-30))) - center) / xmax
    pos = alpha * np.tanh((gamma * x) / alpha)
    neg = beta * np.tanh((delta * x) / beta)
    return (np.where(x >= 0, pos, neg) * sigma).astype(np.float32)


def balanced_draws(
    samples: np.ndarray, heldout_ids: np.ndarray, k: int, draws_per_cosmology: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select identical draw indices for every held-out cosmology."""
    maps = as_maps(samples, "generated samples")
    heldout = np.asarray(heldout_ids, dtype=np.int64)
    if heldout.ndim != 1 or len(np.unique(heldout)) != len(heldout):
        raise ValueError("held-out simulation IDs must be a unique vector")
    if k < 1 or len(maps) != len(heldout) * k:
        raise ValueError("generated sample count disagrees with heldout IDs and k")
    if not 1 <= draws_per_cosmology <= k:
        raise ValueError("draws_per_cosmology must be between 1 and k")
    draw_indices = np.linspace(0, k - 1, draws_per_cosmology, dtype=np.int64)
    flat = (np.arange(len(heldout))[:, None] * k + draw_indices[None, :]).ravel()
    return maps[flat], np.repeat(heldout, len(draw_indices)), np.tile(draw_indices, len(heldout))


def nearest_centered_pixel_cosine(
    queries: np.ndarray,
    reference: np.ndarray,
    *,
    reference_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact maximum mean-centered pixel cosine over *all* reference maps.

    Query/reference chunks limit memory. This is a spatially aligned pixel
    comparison, not SSCD or translation-invariant feature similarity.
    """
    query = as_maps(queries, "queries")
    ref = as_maps(reference, "reference")
    if query.shape[1:] != ref.shape[1:]:
        raise ValueError("query and reference map shapes disagree")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    def unit_rows(images: np.ndarray) -> np.ndarray:
        flat = np.asarray(images, dtype=np.float32).reshape(len(images), -1).copy()
        if not np.isfinite(flat).all():
            raise ValueError("cosine inputs contain non-finite values")
        flat -= flat.mean(axis=1, keepdims=True)
        length = np.linalg.norm(flat, axis=1)
        if np.any(length <= 0):
            raise ValueError("cosine input contains a constant map")
        flat /= length[:, None]
        return flat

    q = unit_rows(query)
    best = np.full(len(q), -np.inf, dtype=np.float32)
    indices = np.full(len(q), -1, dtype=np.int64)
    for start in range(0, len(ref), chunk_size):
        raw_chunk = ref[start : start + chunk_size]
        chunk = reference_transform(raw_chunk) if reference_transform else raw_chunk
        r = unit_rows(chunk)
        cosine = q @ r.T
        local = np.argmax(cosine, axis=1)
        value = cosine[np.arange(len(q)), local]
        improves = value > best
        best[improves] = value[improves]
        indices[improves] = start + local[improves]
    return best, indices


def radial_power_batch(images: np.ndarray, edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return mean-subtracted 2D FFT power per map in common radial k bins."""
    maps = as_maps(images, "power-spectrum maps").astype(np.float64)
    height, width = maps.shape[1:]
    if height != width:
        raise ValueError("power-spectrum maps must be square")
    if edges is None:
        edges = np.linspace(1.0, height / 2, 26)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("radial k edges must be strictly increasing")
    freq = np.fft.fftfreq(height) * height
    k = np.hypot(freq[:, None], freq[None, :])
    bins = np.searchsorted(edges, k, side="right") - 1
    valid = (bins >= 0) & (bins < len(edges) - 1)
    counts = np.bincount(bins[valid].ravel(), minlength=len(edges) - 1)
    if np.any(counts == 0):
        raise ValueError("one or more radial k bins contain no Fourier modes")
    output = np.empty((len(maps), len(edges) - 1), dtype=np.float64)
    for row, image in enumerate(maps):
        centered = image - image.mean()
        power = np.abs(np.fft.fft2(centered)) ** 2 / image.size
        sums = np.bincount(bins[valid].ravel(), weights=power[valid].ravel(), minlength=len(counts))
        output[row] = sums / counts
    return output, 0.5 * (edges[:-1] + edges[1:])


def compare_power(generated: np.ndarray, real: np.ndarray, edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ratio of ensemble mean model-space power; no per-map ratio averaging."""
    gen_power, centers = radial_power_batch(generated, edges)
    real_power, real_centers = radial_power_batch(real, edges)
    if not np.array_equal(centers, real_centers):
        raise ValueError("inconsistent k binning")
    denominator = real_power.mean(axis=0)
    return gen_power.mean(axis=0) / np.maximum(denominator, 1e-30), centers
