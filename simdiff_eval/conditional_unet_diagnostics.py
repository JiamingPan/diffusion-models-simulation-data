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


def pairwise_centered_cosine_stats(maps: np.ndarray) -> dict:
    """All distinct unordered pairs; diagonal/self-pairs are excluded."""
    x = as_maps(maps, "pairwise maps").astype(np.float64).reshape(len(maps), -1)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("pairwise comparison requires at least two finite maps")
    x -= x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("pairwise comparison contains a constant map")
    x /= norms[:, None]
    values = np.clip((x @ x.T)[np.triu_indices(len(x), 1)], -1, 1)
    return {"mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "near_duplicate_fraction": float(np.mean(values > .99)),
            "n_pairs": int(len(values))}


def maps_per_cosmology(simulation_ids: np.ndarray) -> dict:
    ids = np.asarray(simulation_ids)
    if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("simulation IDs must be a nonempty integer vector")
    unique, counts = np.unique(ids, return_counts=True)
    return {"mean": float(counts.mean()), "max": int(counts.max()),
            "min": int(counts.min()), "n_cosmologies": len(unique),
            "n_maps": len(ids), "simulation_ids": unique, "counts": counts}


def parameter_nearest_basis_indices(train_theta: np.ndarray, simulation_ids: np.ndarray,
                                    requested: np.ndarray, scales: np.ndarray,
                                    k_cosmologies: int = 16, cap: int = 64) -> dict:
    """Choose parameter-nearest cosmologies, then round-robin their ordered map rows.

    The cap preserves at least one row of every selected cosmology. Selection never
    uses image similarity, a fitted weight or probe recovery.
    """
    theta = np.asarray(train_theta, dtype=float)
    ids = np.asarray(simulation_ids)
    target, scale = np.asarray(requested, dtype=float), np.asarray(scales, dtype=float)
    counts = maps_per_cosmology(ids)
    if theta.shape != (len(ids), 6) or target.shape != (6,) or scale.shape != (6,):
        raise ValueError("parameter basis requires (N,6), (N,), (6,), (6,) inputs")
    if not all(np.isfinite(a).all() for a in (theta, target, scale)) or np.any(scale <= 0):
        raise ValueError("parameters and positive scales must be finite")
    if k_cosmologies < 1 or cap < k_cosmologies:
        raise ValueError("basis cap must be at least the positive cosmology count")
    unique = counts["simulation_ids"]
    groups = [np.flatnonzero(ids == sim) for sim in unique]
    first = np.array([g[0] for g in groups])
    for group in groups:
        if not np.allclose(theta[group], theta[group[0]], rtol=0, atol=1e-6):
            raise ValueError("one training simulation has inconsistent parameter labels")
    distances = np.linalg.norm((theta[first]-target)/scale, axis=1)
    ordered = np.argsort(distances, kind="stable")[:k_cosmologies]
    chosen = [groups[i] for i in ordered]
    rows = []
    for offset in range(max(map(len, chosen))):
        rows.extend(int(group[offset]) for group in chosen if offset < len(group))
        if len(rows) >= cap:
            break
    return {"indices": np.asarray(rows[:cap], dtype=int), "cosmology_ids": unique[ordered],
            "distances": distances[ordered]}


def nnls_blend_fit(target: np.ndarray, basis: np.ndarray) -> dict:
    """Nonnegative centered linear fit, NOT a sum-to-one constrained mixture.

    R2 = 1 - squared residual / centered target energy. Re-add the target mean
    for display/probe inputs. Return raw coefficient sum to expose rescaling;
    normalized weights are used only to summarize component participation.
    """
    from scipy.optimize import nnls

    b = as_maps(basis, "NNLS basis").astype(np.float64)
    t = np.asarray(target, dtype=np.float64)
    if t.ndim != 2 or b.shape[1:] != t.shape:
        raise ValueError("NNLS target must match basis image shape")
    if not np.isfinite(t).all() or not np.isfinite(b).all():
        raise ValueError("NNLS inputs must be finite")
    y = (t-t.mean()).ravel()
    a = b.reshape(len(b), -1).T.copy()
    a -= a.mean(axis=0, keepdims=True)
    energy = float(y @ y)
    if energy == 0 or np.any(np.linalg.norm(a, axis=0) == 0):
        raise ValueError("NNLS requires nonconstant target and basis maps")
    weights, _ = nnls(a, y, maxiter=max(1000, 3*len(b)))
    prediction = a @ weights
    total = float(weights.sum())
    normalized = weights/total if total > 0 else np.zeros_like(weights)
    effective = 1/float(normalized @ normalized) if total > 0 else 0.0
    return {"weights": weights, "normalized_weights": normalized, "weight_sum": total,
            "r2": float(1-np.sum((y-prediction)**2)/energy),
            "effective_components": effective,
            "top_index": int(np.argmax(weights)) if total > 0 else -1,
            "reconstruction": prediction.reshape(t.shape)+t.mean()}


def mean_preserving_blend(original: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Equal pixel average in model space; restore original's map mean."""
    original = np.asarray(original, dtype=float)
    other = as_maps(neighbors, "blend neighbors").astype(float)
    if original.ndim != 2 or other.shape[1:] != original.shape:
        raise ValueError("blend images must have matching shapes")
    if not np.isfinite(original).all() or not np.isfinite(other).all():
        raise ValueError("blend images must be finite")
    blended = (original + other.sum(axis=0))/(len(other)+1)
    return blended-blended.mean()+original.mean()


def lookup_blend_run(samples: np.ndarray, training_raw: np.ndarray, train_theta: np.ndarray,
                     simulation_ids: np.ndarray, heldout_ids: np.ndarray, requested: np.ndarray,
                     scales: np.ndarray, real_model_maps: np.ndarray,
                     normalization: dict, k: int = 64) -> dict:
    """Array-only driver for one N, with fresh per-run inputs (no notebook loop state).

    real_model_maps has shape (cosmologies, k, H, W), ordered by evenly spaced
    z slices. Fits use first/last draws and the matching first/last real slices.
    """
    gen = as_maps(samples, "generated maps")
    train = as_maps(training_raw, "training maps")
    heldout, ids = np.asarray(heldout_ids), np.asarray(simulation_ids)
    truth, real = np.asarray(requested, dtype=float), np.asarray(real_model_maps)
    counts = maps_per_cosmology(ids)
    if k < 2 or gen.shape != (len(heldout)*k, *train.shape[1:]):
        raise ValueError("generated maps do not match cosmology-major layout")
    if real.shape != (len(heldout), k, *train.shape[1:]) or truth.shape != (len(heldout), 6):
        raise ValueError("real slices or requested parameters have mismatched shapes")
    if len(ids) != len(train) or len(np.unique(heldout)) != len(heldout):
        raise ValueError("training row IDs or held-out IDs mismatch")
    if np.intersect1d(ids, heldout).size:
        raise ValueError("training and held-out simulations overlap")
    diversity, fits, reconstructions, two_blends, three_blends = [], [], [], [], []
    example = None
    for i, sim in enumerate(heldout):
        g = gen[i*k:(i+1)*k]
        for kind, maps in [("generated", g), ("real slices (correlated)", real[i])]:
            diversity.append(dict(heldout_sim=int(sim), kind=kind,
                                  **pairwise_centered_cosine_stats(maps)))
        selected = parameter_nearest_basis_indices(train_theta, ids, truth[i], scales)
        indices = selected["indices"]
        basis = normalize_raw_hi(train[indices], normalization)
        for draw in [0, k-1]:
            for kind, target in [("generated", g[draw]), ("held-out real", real[i, draw])]:
                fit = nnls_blend_fit(target, basis)
                top_sim = int(ids[indices[fit["top_index"]]]) if fit["top_index"] >= 0 else -1
                fits.append({"heldout_sim": int(sim), "draw": draw, "kind": kind,
                             "r2": fit["r2"], "effective_components": fit["effective_components"],
                             "weight_sum": fit["weight_sum"], "basis_size": len(basis),
                             "top_is_nearest": top_sim == int(selected["cosmology_ids"][0])})
                if kind == "generated" and draw == 0:
                    reconstructions.append(fit["reconstruction"])
                    if example is None:
                        order = np.argsort(-fit["weights"], kind="stable")[:3]
                        example = {"generated": target.copy(), "reconstruction": fit["reconstruction"],
                                   "residual": target-fit["reconstruction"], "basis": basis[order],
                                   "weights": fit["normalized_weights"][order],
                                   "weight_sum": fit["weight_sum"], "rows": indices[order],
                                   "r2": fit["r2"], "heldout_sim": int(sim)}
        if len(selected["cosmology_ids"]) < 2:
            raise ValueError("blending test requires at least two training cosmologies")
        # Round-robin basis starts with one selected training slice per cosmology.
        two_blends.append(mean_preserving_blend(real[i, 0], basis[:1]))
        three_blends.append(mean_preserving_blend(real[i, 0], basis[:2]))
    return {"diversity": diversity, "fits": fits, "counts": counts, "example": example,
            "first_last_allclose": bool(np.allclose(gen[0], gen[k-1], rtol=0, atol=1e-4)),
            "first_last_max_abs": float(np.max(np.abs(gen[0]-gen[k-1]))),
            "probe_inputs": {"Real held-out z=0": real[:, 0].copy(),
                             "2-map blend": np.asarray(two_blends),
                             "3-map blend": np.asarray(three_blends),
                             "NNLS reconstruction": np.asarray(reconstructions)},
            "probe_requested": truth[:, 0].copy(), "heldout_ids": heldout.copy()}


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


def recovery_interval_coverage(points, predictions=None):
    """Coverage across held-out simulations, never across individual draws.

    Points supply the saved 16/84 percentiles. Per-draw predictions additionally
    allow a full central-interval coverage curve and 95% inclusion. These are
    recovery-ensemble intervals, not independently validated posteriors.
    """
    import pandas as pd

    keys = ["run_name", "dataset_size", "heldout_sim", "parameter"]
    numeric = ["theta_in", "theta_rec_median", "theta_rec_q16", "theta_rec_q84", "n_samples"]
    missing = set(keys + numeric) - set(points.columns)
    if missing:
        raise ValueError(f"Recovery points missing columns: {sorted(missing)}")
    if points.empty or points.duplicated(keys).any():
        raise ValueError("Recovery points are empty or duplicate a simulation/parameter")
    if not np.isfinite(points[numeric].to_numpy(float)).all():
        raise ValueError("Non-finite recovery point values")
    if ((points.theta_rec_q16 > points.theta_rec_median)
            | (points.theta_rec_median > points.theta_rec_q84)).any():
        raise ValueError("Recovery quantiles are not ordered")
    intervals = []
    if predictions is None:
        for _, row in points.iterrows():
            record = {key: row[key] for key in keys}
            intervals.append(dict(record, nominal=0.68,
                covered=bool(row.theta_rec_q16 <= row.theta_in <= row.theta_rec_q84),
                width=float(row.theta_rec_q84-row.theta_rec_q16)))
    else:
        required = keys + ["seed_index", "theta_in", "theta_rec"]
        if set(required) - set(predictions.columns):
            raise ValueError("Per-draw recovery table is missing required columns")
        if predictions.duplicated(keys + ["seed_index"]).any():
            raise ValueError("Duplicate per-draw recovery prediction")
        if not np.isfinite(predictions[["theta_in", "theta_rec"]].to_numpy(float)).all():
            raise ValueError("Non-finite per-draw predictions")
        indexed = points.set_index(keys)
        grouped = predictions.groupby(keys, sort=False)
        if set(grouped.groups) != set(indexed.index):
            raise ValueError("Per-draw predictions and recovery points name different simulations/runs")
        levels = sorted(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95]))
        for key, group in grouped:
            point = indexed.loc[key]
            values = group.theta_rec.to_numpy(float)
            if len(values) < 2 or len(values) != point.n_samples:
                raise ValueError(f"Unexpected recovery draw count for {key}")
            if not np.allclose(group.theta_in, point.theta_in, rtol=0, atol=1e-6):
                raise ValueError(f"Requested parameters differ between recovery tables for {key}")
            quantiles = np.quantile(values, [0.16, 0.5, 0.84])
            if not np.allclose(quantiles, point[["theta_rec_q16", "theta_rec_median", "theta_rec_q84"]].to_numpy(float), rtol=1e-5, atol=1e-7):
                raise ValueError(f"Per-draw predictions do not reproduce saved quantiles for {key}")
            for level in levels:
                low, high = np.quantile(values, [(1-level)/2, (1+level)/2])
                intervals.append(dict(zip(keys, key), nominal=level,
                    covered=bool(low <= point.theta_in <= high), width=float(high-low)))
    individual = pd.DataFrame(intervals)
    groups = ["run_name", "dataset_size", "parameter", "nominal"]
    curve = individual.groupby(groups, as_index=False).agg(
        empirical=("covered", "mean"), n_cosmologies=("heldout_sim", "nunique"),
        covered_cosmologies=("covered", "sum"), median_width=("width", "median"))
    # Wilson 95% binomial interval: the unit is a held-out simulation, not a draw.
    p = curve.empirical.to_numpy(float)
    n = curve.n_cosmologies.to_numpy(float)
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    curve["ci_low"] = np.maximum(0, center-half)
    curve["ci_high"] = np.minimum(1, center+half)
    return curve
