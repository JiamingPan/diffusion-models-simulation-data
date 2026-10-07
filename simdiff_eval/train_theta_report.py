"""Pure NumPy report for the training-theta test (conditional UNet sampled at its
own training cosmologies). Draws are aggregated within each cosmology first; the
cosmologies are the sampling units for every summary.

A cosmology may own several training maps (multiplicity m > 1). "Own" statistics
then use the best-matching map of that cosmology, and the real-map probe control
is the median over that cosmology's training maps."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .conditional_unet_diagnostics import as_maps, pairwise_centered_cosine_stats


def _unit_rows(maps: np.ndarray) -> np.ndarray:
    x = as_maps(maps, "maps").astype(np.float64).reshape(len(maps), -1)
    if not np.isfinite(x).all():
        raise ValueError("maps must be finite")
    x = x - x.mean(axis=1, keepdims=True)
    n = np.linalg.norm(x, axis=1)
    if np.any(n == 0):
        raise ValueError("constant map")
    return x / n[:, None]


def summarize(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "q16": float(np.quantile(v, .16)),
            "q84": float(np.quantile(v, .84)), "mean": float(v.mean()), "n": int(len(v))}


def train_theta_report(samples: np.ndarray, train_maps: np.ndarray, n_seeds: int,
                       true_omega: np.ndarray,
                       predict_omega: Callable[[np.ndarray], np.ndarray] | None = None,
                       group_index: np.ndarray | None = None) -> dict:
    """samples: (C*n_seeds, H, W) cosmology-major; train_maps: (N, H, W) model-space
    training maps; group_index: (N,) cosmology index 0..C-1 of each training map
    (None = one map per cosmology, in order); true_omega: (C,) requested Omega_m.
    predict_omega maps (n, H, W) model-space maps -> (n,) recovered Omega_m."""
    gen = as_maps(samples, "generated")
    train = as_maps(train_maps, "training maps")
    truth = np.asarray(true_omega, dtype=float)
    groups = np.arange(len(train)) if group_index is None else np.asarray(group_index, dtype=int)
    if groups.shape != (len(train),) or groups.min() < 0:
        raise ValueError("group_index must give one cosmology index per training map")
    c = int(groups.max()) + 1
    if n_seeds < 1 or gen.shape != (c * n_seeds, *train.shape[1:]) or truth.shape != (c,):
        raise ValueError("samples must be cosmology-major with n_seeds draws per cosmology")
    if len(np.unique(groups)) != c:
        raise ValueError("every cosmology index must own at least one training map")
    g, t = _unit_rows(gen), _unit_rows(train)
    cos_all = g @ t.T                                   # (C*n_seeds, N)
    draw_group = np.repeat(np.arange(c), n_seeds)
    own_mask = groups[None, :] == draw_group[:, None]   # which training maps belong to the draw's cosmology
    cos_own = np.where(own_mask, cos_all, -np.inf).max(axis=1)
    cos_max = cos_all.max(axis=1)
    nearest_group = groups[cos_all.argmax(axis=1)]
    per_cosmology = []
    for i in range(c):
        sl = slice(i * n_seeds, (i + 1) * n_seeds)
        per_cosmology.append({
            "cosmology_index": i, "n_training_maps": int(np.sum(groups == i)),
            "cos_to_own_map": float(cos_own[sl].mean()),
            "max_cos_any_training_map": float(cos_max[sl].mean()),
            "nearest_is_own_fraction": float(np.mean(nearest_group[sl] == i)),
            "within_cosmology_draw_cosine": float(pairwise_centered_cosine_stats(gen[sl])["mean"]) if n_seeds > 1 else np.nan,
        })
    report = {"n_cosmologies": c, "n_seeds": n_seeds, "n_fields": int(len(gen)),
              "n_training_maps": int(len(train)), "per_cosmology": per_cosmology}
    keys = ["cos_to_own_map", "max_cos_any_training_map", "nearest_is_own_fraction"]
    if n_seeds > 1:
        keys.append("within_cosmology_draw_cosine")
    report["summary"] = {k: summarize([p[k] for p in per_cosmology]) for k in keys}
    report["summary"]["all_draws_nearest_is_own_fraction"] = float(np.mean(nearest_group == draw_group))
    if predict_omega is not None:
        pred_gen = np.asarray(predict_omega(gen), dtype=float)
        pred_real = np.asarray(predict_omega(train), dtype=float)
        if pred_gen.shape != (len(gen),) or pred_real.shape != (len(train),) or not np.isfinite(pred_gen).all() or not np.isfinite(pred_real).all():
            raise ValueError("probe returned invalid Omega_m predictions")
        gen_by = pred_gen.reshape(c, n_seeds)
        gen_median = np.median(gen_by, axis=1)
        real_median = np.array([np.median(pred_real[groups == i]) for i in range(c)])
        for i, p in enumerate(per_cosmology):
            p.update({"true_omega": float(truth[i]),
                      "generated_omega_median": float(gen_median[i]),
                      "generated_omega_q16": float(np.quantile(gen_by[i], .16)),
                      "generated_omega_q84": float(np.quantile(gen_by[i], .84)),
                      "real_omega_median": float(real_median[i]),
                      "generated_bias": float(gen_median[i] - truth[i]),
                      "real_bias": float(real_median[i] - truth[i]),
                      "paired_generated_minus_real": float(gen_median[i] - real_median[i])})
        report["summary"]["generated_bias"] = summarize(gen_median - truth)
        report["summary"]["real_training_map_bias"] = summarize(real_median - truth)
        report["summary"]["paired_generated_minus_real"] = summarize(gen_median - real_median)
        report["summary"]["within_cosmology_omega_spread_q84_minus_q16"] = summarize(
            np.quantile(gen_by, .84, axis=1) - np.quantile(gen_by, .16, axis=1))
    return report
