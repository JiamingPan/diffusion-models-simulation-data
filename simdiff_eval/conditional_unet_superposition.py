"""Superposition tests for low-N conditional UNet samples.

Everything here is read-only NumPy/SciPy on saved arrays. The tests ask a narrower
question than "is the model a blend": how much of a generated field is explained
by (a) one, two, three or all parameter-nearest training maps, (b) random training
maps that exclude the parameter-nearest ones, (c) only the large scales, and
(d) the two endpoint training maps of a straight line in parameter space.
Predictions are printed by the notebook, never assumed here.
"""

from __future__ import annotations

import numpy as np

from .conditional_unet_diagnostics import (
    as_maps,
    nnls_blend_fit,
    normalize_raw_hi,
    pairwise_centered_cosine_stats,
    parameter_nearest_basis_indices,
)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-centered pixel cosine between two 2-D maps."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("cosine inputs must be finite maps of equal shape")
    x = x - x.mean()
    y = y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("cosine input contains a constant map")
    return float(np.clip(x @ y / (nx * ny), -1, 1))


def lowpass_maps(maps: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Periodic Gaussian low-pass in pixel units (maps are periodic boxes)."""
    from scipy.ndimage import gaussian_filter

    x = as_maps(maps, "low-pass maps").astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("low-pass input must be finite")
    if sigma_pixels < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma_pixels == 0:
        return x.copy()
    return np.stack([gaussian_filter(m, sigma_pixels, mode="wrap") for m in x])


def random_basis_indices(n_train: int, exclude: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Uniform random training rows that avoid `exclude` (the parameter-nearest rows)."""
    exclude = np.asarray(exclude, dtype=int)
    pool = np.setdiff1d(np.arange(int(n_train)), exclude)
    if size < 1 or len(pool) < 1:
        raise ValueError("no non-excluded training rows are available for the random basis")
    rng = np.random.default_rng(int(seed))
    # If fewer rows remain than requested (tiny N), use all of them; basis_size is reported.
    return np.sort(rng.choice(pool, size=int(min(size, len(pool))), replace=False))


# ----------------------------------------------------------------------------
# Test A: how many parameter-nearest maps are actually needed, plus controls
# ----------------------------------------------------------------------------

def nnls_topk_curve(target: np.ndarray, ordered_basis: np.ndarray,
                    ks: tuple[int, ...] = (1, 2, 3)) -> list[dict]:
    """R2 when only the first k basis maps (already ordered by parameter distance)
    are offered to NNLS, plus the full basis. Reports the single-map cos^2 too, so the
    k=1 row can be compared with the nearest-training cosine of the notebook."""
    basis = as_maps(ordered_basis, "ordered basis")
    rows = []
    wanted = sorted({int(k) for k in ks if 1 <= int(k) <= len(basis)} | {len(basis)})
    for k in wanted:
        fit = nnls_blend_fit(target, basis[:k])
        rows.append({"k": k, "r2": fit["r2"], "weight_sum": fit["weight_sum"],
                     "effective_components": fit["effective_components"]})
    rows[0]["single_map_cos2"] = centered_cosine(target, basis[0]) ** 2
    return rows


def superposition_controls_run(
    samples: np.ndarray, training_raw: np.ndarray, train_theta: np.ndarray,
    simulation_ids: np.ndarray, heldout_ids: np.ndarray, requested: np.ndarray,
    scales: np.ndarray, real_model_maps: np.ndarray, normalization: dict, *,
    k: int = 64, k_cosmologies: int = 16, cap: int = 64, sigmas: tuple[float, ...] = (2.0, 4.0),
    seed: int = 123, draws: tuple[int, ...] = (0,),
) -> list[dict]:
    """Per held-out cosmology and draw: top-k curve, random-basis control and
    low-pass R2, for the generated map and for a real slice at the same cosmology.

    `real_model_maps` has shape (cosmologies, k, H, W) in model space (as in
    lookup_blend_run). Generated samples are cosmology-major with k draws each.
    """
    gen = as_maps(samples, "generated maps")
    train = as_maps(training_raw, "training maps")
    heldout, ids = np.asarray(heldout_ids), np.asarray(simulation_ids)
    truth, real = np.asarray(requested, dtype=float), np.asarray(real_model_maps)
    if gen.shape != (len(heldout) * k, *train.shape[1:]):
        raise ValueError("generated maps do not match cosmology-major layout")
    if real.shape != (len(heldout), k, *train.shape[1:]) or truth.shape != (len(heldout), 6):
        raise ValueError("real slices or requested parameters have mismatched shapes")
    if len(ids) != len(train):
        raise ValueError("training row IDs mismatch")
    rows = []
    for i, sim in enumerate(heldout):
        selected = parameter_nearest_basis_indices(train_theta, ids, truth[i], scales,
                                                   k_cosmologies=k_cosmologies, cap=cap)
        near_idx = selected["indices"]
        near_basis = normalize_raw_hi(train[near_idx], normalization)
        rand_idx = random_basis_indices(len(train), near_idx, len(near_idx), seed + int(sim))
        rand_basis = normalize_raw_hi(train[rand_idx], normalization)
        for draw in draws:
            for kind, target in [("generated", gen[i * k + draw]), ("held-out real", real[i, draw])]:
                base = {"heldout_sim": int(sim), "draw": int(draw), "kind": kind}
                for row in nnls_topk_curve(target, near_basis):
                    rows.append(dict(base, test="parameter-nearest top-k", k=row["k"], sigma=0.0,
                                     r2=row["r2"], weight_sum=row["weight_sum"],
                                     effective_components=row["effective_components"],
                                     single_map_cos2=row.get("single_map_cos2", np.nan)))
                fit = nnls_blend_fit(target, rand_basis)
                rows.append(dict(base, test="random basis (excludes nearest)", k=len(rand_basis),
                                 sigma=0.0, r2=fit["r2"], weight_sum=fit["weight_sum"],
                                 effective_components=fit["effective_components"],
                                 single_map_cos2=np.nan))
                for sigma in sigmas:
                    smooth_target = lowpass_maps(target[None], sigma)[0]
                    smooth_basis = lowpass_maps(near_basis, sigma)
                    fit = nnls_blend_fit(smooth_target, smooth_basis)
                    rows.append(dict(base, test="low-pass parameter-nearest", k=len(smooth_basis),
                                     sigma=float(sigma), r2=fit["r2"], weight_sum=fit["weight_sum"],
                                     effective_components=fit["effective_components"],
                                     single_map_cos2=centered_cosine(smooth_target, smooth_basis[0]) ** 2))
    return rows


# ----------------------------------------------------------------------------
# Test B: straight lines in parameter space between two training cosmologies
# ----------------------------------------------------------------------------

def mutually_nearest_pairs(train_theta: np.ndarray, simulation_ids: np.ndarray,
                           scales: np.ndarray, n_pairs: int) -> list[dict]:
    """Pairs (A,B) of distinct training cosmologies that are each other's nearest
    neighbour in scaled six-parameter distance, sorted by distance. One map row per
    cosmology (the first row) is returned so the endpoint images are well defined."""
    theta = np.asarray(train_theta, dtype=float)
    ids = np.asarray(simulation_ids)
    scale = np.asarray(scales, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 6 or len(ids) != len(theta) or scale.shape != (6,):
        raise ValueError("expected (N,6) parameters, (N,) ids and (6,) scales")
    if np.any(scale <= 0) or not np.isfinite(theta).all():
        raise ValueError("parameters and positive scales must be finite")
    unique, first = np.unique(ids, return_index=True)
    if len(unique) < 2:
        raise ValueError("need at least two training cosmologies")
    z = theta[first] / scale
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)
    pairs = []
    for a in range(len(unique)):
        b = int(nn[a])
        if nn[b] == a and a < b:
            pairs.append({"sim_a": int(unique[a]), "sim_b": int(unique[b]),
                          "row_a": int(first[a]), "row_b": int(first[b]),
                          "distance": float(d[a, b])})
    pairs.sort(key=lambda p: p["distance"])
    if len(pairs) < n_pairs:
        raise ValueError(f"only {len(pairs)} mutually nearest pairs exist; asked for {n_pairs}")
    return pairs[:n_pairs]


def interpolation_labels(train_theta_raw: np.ndarray, pairs: list[dict],
                         s_values: np.ndarray, n_seeds: int) -> tuple[np.ndarray, list[dict]]:
    """Raw-parameter labels, pair-major then s then seed, plus a row-level index."""
    theta = np.asarray(train_theta_raw, dtype=float)
    s_values = np.asarray(s_values, dtype=float)
    if n_seeds < 1 or s_values.ndim != 1 or len(s_values) == 0:
        raise ValueError("need at least one seed and one s value")
    labels, index = [], []
    for p, pair in enumerate(pairs):
        a, b = theta[pair["row_a"]], theta[pair["row_b"]]
        for si, s in enumerate(s_values):
            value = (1.0 - s) * a + s * b
            for seed in range(n_seeds):
                labels.append(value)
                index.append({"pair": p, "s_index": si, "s": float(s), "seed": seed,
                              "row": len(labels) - 1})
    return np.asarray(labels, dtype=np.float32), index


def theta_interpolation_analysis(samples: np.ndarray, endpoint_a: np.ndarray,
                                 endpoint_b: np.ndarray, index: list[dict],
                                 s_values: np.ndarray) -> list[dict]:
    """Per (pair, s): cosine to each endpoint, two-map NNLS weights and R2,
    and draw-to-draw cosine over the seeds. Endpoints are model-space maps,
    one per pair, in the same pair order as `index`."""
    gen = as_maps(samples, "interpolation samples")
    a_maps, b_maps = as_maps(endpoint_a, "endpoint A"), as_maps(endpoint_b, "endpoint B")
    if len(gen) != len(index):
        raise ValueError("sample count does not match the label index")
    n_pairs = len(a_maps)
    if len(b_maps) != n_pairs or a_maps.shape[1:] != gen.shape[1:]:
        raise ValueError("endpoint maps must match sample shape and pair count")
    rows = []
    for p in range(n_pairs):
        for si, s in enumerate(np.asarray(s_values, dtype=float)):
            members = [r["row"] for r in index if r["pair"] == p and r["s_index"] == si]
            if not members:
                raise ValueError(f"no samples for pair {p}, s index {si}")
            draws = gen[members]
            basis = np.stack([a_maps[p], b_maps[p]])
            per_draw = [nnls_blend_fit(d, basis) for d in draws]
            w = np.array([f["normalized_weights"] for f in per_draw])
            diversity = pairwise_centered_cosine_stats(draws)["mean"] if len(draws) > 1 else np.nan
            rows.append({"pair": p, "s": float(s),
                         "cos_a": float(np.mean([centered_cosine(d, a_maps[p]) for d in draws])),
                         "cos_b": float(np.mean([centered_cosine(d, b_maps[p]) for d in draws])),
                         "weight_a": float(w[:, 0].mean()), "weight_b": float(w[:, 1].mean()),
                         "weight_sum": float(np.mean([f["weight_sum"] for f in per_draw])),
                         "two_map_r2": float(np.mean([f["r2"] for f in per_draw])),
                         "draw_to_draw_cosine": float(diversity), "n_draws": len(draws)})
    return rows


def training_theta_analysis(samples: np.ndarray, own_maps: np.ndarray, n_seeds: int) -> list[dict]:
    """Samples requested at the training cosmologies themselves, cosmology-major with
    `n_seeds` draws each; `own_maps[i]` is the training map of cosmology i."""
    gen = as_maps(samples, "training-theta samples")
    own = as_maps(own_maps, "own training maps")
    if n_seeds < 1 or len(gen) != len(own) * n_seeds or gen.shape[1:] != own.shape[1:]:
        raise ValueError("training-theta samples must be cosmology-major with n_seeds draws")
    rows = []
    for i in range(len(own)):
        draws = gen[i * n_seeds:(i + 1) * n_seeds]
        rows.append({"cosmology_index": i,
                     "cos_to_own_map": float(np.mean([centered_cosine(d, own[i]) for d in draws])),
                     "draw_to_draw_cosine": float(pairwise_centered_cosine_stats(draws)["mean"])
                     if n_seeds > 1 else np.nan})
    return rows
