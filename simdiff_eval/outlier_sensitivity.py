"""Conservative helpers for post-hoc outlier sensitivity analyses."""

from __future__ import annotations

from typing import Any

import numpy as np


def novelty_bounds_after_filtering(
    *,
    n_total: int,
    n_removed: int,
    novelty_score: float,
) -> dict[str, Any]:
    """Bound retained-sample novelty when removed copy labels are unavailable.

    Aggregate novelty determines the total number of copy-like samples, but it
    does not identify which sample indices are copy-like. The returned interval
    covers the two extrema: every removed sample was copy-like, or every removed
    sample was novel, subject to the aggregate copy count.
    """

    n_total = int(n_total)
    n_removed = int(n_removed)
    novelty_score = float(novelty_score)
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    if n_removed < 0 or n_removed >= n_total:
        raise ValueError("n_removed must be in [0, n_total)")
    if not np.isfinite(novelty_score) or not 0.0 <= novelty_score <= 1.0:
        raise ValueError("novelty_score must be finite and in [0, 1]")

    n_kept = n_total - n_removed
    n_copies_total = int(np.clip(np.rint((1.0 - novelty_score) * n_total), 0, n_total))
    copies_kept_min = max(0, n_copies_total - n_removed)
    copies_kept_max = min(n_copies_total, n_kept)
    novelty_lower = 1.0 - copies_kept_max / n_kept
    novelty_upper = 1.0 - copies_kept_min / n_kept
    return {
        "n_total": n_total,
        "n_removed": n_removed,
        "n_kept": n_kept,
        "n_copies_total": n_copies_total,
        "copies_kept_min": copies_kept_min,
        "copies_kept_max": copies_kept_max,
        "novelty_lower": novelty_lower,
        "novelty_upper": novelty_upper,
    }


def _validated_keep_mask(length: int, keep_mask: np.ndarray) -> np.ndarray:
    keep = np.asarray(keep_mask, dtype=bool)
    if keep.ndim != 1 or len(keep) != int(length):
        raise ValueError(f"keep_mask must have shape ({length},)")
    if not np.any(keep):
        raise ValueError("keep_mask excludes every sample")
    return keep


def filtered_histogram_probability(
    samples: np.ndarray,
    *,
    bins: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    """Return a normalized pixel histogram for retained generated samples."""

    values = np.asarray(samples)
    if values.ndim < 2:
        raise ValueError("samples must have a sample axis and at least one value axis")
    keep = _validated_keep_mask(len(values), keep_mask)
    counts = np.histogram(values[keep], bins=np.asarray(bins, dtype=float))[0].astype(float)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("retained samples have no values inside the histogram bins")
    return counts / total


def filtered_power_summary(
    power_ratios: np.ndarray,
    *,
    keep_mask: np.ndarray,
) -> dict[str, Any]:
    """Summarize per-sample power ratios after applying a fixed sample mask."""

    ratios = np.asarray(power_ratios, dtype=float)
    if ratios.ndim != 2:
        raise ValueError("power_ratios must have shape (samples, k_bins)")
    keep = _validated_keep_mask(len(ratios), keep_mask)
    retained = ratios[keep]
    mean = np.nanmean(retained, axis=0)
    median = np.nanmedian(retained, axis=0)
    variance = np.nanvar(retained, axis=0)
    valid = np.isfinite(mean) & (mean > 0)
    if not np.any(valid):
        raise ValueError("retained power ratios have no finite positive mean bins")
    return {
        "mean": mean,
        "median": median,
        "variance": variance,
        "log10_mae": float(np.mean(np.abs(np.log10(mean[valid])))),
        "n_kept": int(keep.sum()),
        "n_removed": int((~keep).sum()),
    }
