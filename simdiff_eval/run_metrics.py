"""Metric functions behind scripts/evaluate_run.py (the ledger's metric source).

Pure numpy/scipy/sklearn; torch only inside the orbit search
(simdiff_eval.orbit_nn). Each function reproduces an existing definition:

* PCA novelty G: notebooks/dit_l16_zero_init_generalization_pca95.ipynb cells 5-7
  (pixel-standardized PCA, randomized solver random_state=0, real-real
  nearest-other quantile threshold, copy iff nearest >= threshold, G = 1 - copy
  fraction). ``orbit="d4"`` adds the max over the 8 D4 elements of each query;
  the real-real threshold then excludes each map's whole orbit.
* P(k) bins and bands: notebooks/pk_bias_conditional_20261006_143831.ipynb
  (25 linear bins over the nonzero |k| range; large kc<8, mid 8-24, small >=24
  up to Nyquist; band value = exp of the mean log ratio).
* One-point PDF L1: simdiff_eval.dit_diagnostics.one_point_l1_common_bins, on
  bins spanning the real maps' range instead of a fixed [-1, 1].
* Patch-mosaic test (ledger unet128_aug_patch_mosaic): p x p patches on a fixed
  grid, best D4 x periodic-window centred cosine against all training maps
  (simdiff_eval.orbit_nn.patch_orbit_max_cosine); excess over held-out real patches
  with a map-bootstrap CI; copy fraction above the train-train (own map excluded) quantile.

All comparisons are in the generator's normalized (log + tanh) model space.
"""
from __future__ import annotations

import numpy as np

# Median 68% width of the evaluation probe (vgg_mlp_encoder) on real held-out maps.
# Source: results/nf_conditional_bias_probe/transform_controls (identity rows); docs/decisions.md 2026-10-06.
W_REAL_OMEGA_M = 0.0931
PK_BAND_NAMES = ("large", "mid", "small")


def as_maps(images, name: str = "maps") -> np.ndarray:
    """(N,H,W) float32 view of (N,H,W) or (N,1,H,W) finite maps."""
    a = np.asarray(images)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 3 or len(a) == 0:
        raise ValueError(f"{name} must have shape (N,H,W) or (N,1,H,W), got {a.shape}")
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contain non-finite values")
    return a


def d4_element(maps: np.ndarray, g: int) -> np.ndarray:
    """Element g of D4 on the last two axes, same order as simdiff_eval.orbit_nn._d4_stack."""
    y = np.rot90(maps, g % 4, axes=(-2, -1))
    return np.ascontiguousarray(y[..., ::-1] if g >= 4 else y)


# ---------------------------------------------------------------- PCA novelty
class PCASpace:
    """Shared PCA embedding; cosine between unit-normalized projections."""

    def __init__(self, fit_maps: np.ndarray, n_components: int = 32, random_state: int = 0):
        from sklearn.decomposition import PCA

        x = as_maps(fit_maps, "PCA fit maps").reshape(len(fit_maps), -1)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        self.pca = PCA(n_components=min(int(n_components), len(x) - 1), svd_solver="randomized",
                       random_state=random_state)
        self.pca.fit((x - self.mean) / self.scale)
        self.n_fit = len(x)

    @property
    def explained_variance(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())

    def project(self, maps: np.ndarray, batch: int = 2048) -> np.ndarray:
        x = as_maps(maps)
        out = []
        for s in range(0, len(x), batch):
            z = self.pca.transform((x[s:s + batch].reshape(-1, x.shape[-1] * x.shape[-2]) - self.mean)
                                   / self.scale).astype(np.float32)
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise ValueError("Zero-norm PCA embedding: cosine is undefined.")
            out.append(z / norms)
        return np.concatenate(out)


def nearest_cosine(query: np.ndarray, reference: np.ndarray, exclude_self: bool = False,
                   block: int = 256) -> np.ndarray:
    """Max cosine of each unit query row against unit reference rows (notebook cell 5)."""
    if exclude_self and len(query) != len(reference):
        raise ValueError("exclude_self needs query and reference to be the same set")
    out = np.empty(len(query), dtype=np.float32)
    for s in range(0, len(query), block):
        scores = query[s:s + block] @ reference.T
        if exclude_self:
            idx = np.arange(len(scores))
            scores[idx, s + idx] = -np.inf
        out[s:s + len(scores)] = scores.max(axis=1)
    return np.clip(out, -1, 1)


def pca_novelty(space: PCASpace, train: np.ndarray, generated: np.ndarray, quantile: float = 0.95,
                orbit: str = "none") -> dict:
    """G = fraction of generated maps whose nearest training map is below the real-real threshold."""
    if orbit not in ("none", "d4"):
        raise ValueError(f"PCA orbit must be 'none' or 'd4', got {orbit!r}")
    train = as_maps(train, "training maps")
    generated = as_maps(generated, "generated maps")
    train_z = space.project(train)
    if orbit == "none":
        real_nn = nearest_cosine(train_z, train_z, exclude_self=True)
        gen_nn = nearest_cosine(space.project(generated), train_z)
    else:
        real_nn = np.full(len(train), -np.inf, dtype=np.float32)
        gen_nn = np.full(len(generated), -np.inf, dtype=np.float32)
        for g in range(8):
            real_nn = np.maximum(real_nn, nearest_cosine(space.project(d4_element(train, g)), train_z,
                                                         exclude_self=True))
            gen_nn = np.maximum(gen_nn, nearest_cosine(space.project(d4_element(generated, g)), train_z))
    threshold = float(np.quantile(real_nn, quantile))
    copying = gen_nn >= threshold
    return {"G": float((~copying).mean()), "copy_fraction": float(copying.mean()), "threshold": threshold,
            "generated_nn_median": float(np.median(gen_nn)), "real_nn_median": float(np.median(real_nn)),
            "generated_nn": gen_nn, "real_nn": real_nn}


# ---------------------------------------------------------------- pixel similarity
def orbit_copy_stats(train: np.ndarray, generated: np.ndarray, heldout: np.ndarray | None,
                     quantile: float = 0.95, max_threshold_queries: int = 2048, device: str | None = None) -> dict:
    """D4 x periodic-shift nearest-training centred cosine (simdiff_eval.orbit_nn).

    Threshold: quantile of train-to-train orbit similarity with each map's whole
    orbit excluded, from at most ``max_threshold_queries`` evenly spaced training
    queries against all training maps.
    """
    from simdiff_eval.orbit_nn import orbit_max_cosine

    train = as_maps(train, "training maps")
    q_idx = np.linspace(0, len(train) - 1, min(len(train), int(max_threshold_queries)), dtype=np.int64)
    ref = orbit_max_cosine(train[q_idx], train, exclude_ref_index=q_idx, device=device)["max_cosine"]
    threshold = float(np.quantile(ref, quantile))
    nn = orbit_max_cosine(as_maps(generated, "generated maps"), train, device=device)
    copies = nn["max_cosine"] > threshold
    transformed = (nn["group_element"] != 0) | (nn["shift_y"] != 0) | (nn["shift_x"] != 0)
    out = {"orbit_threshold": threshold, "orbit_threshold_queries": int(len(q_idx)),
           "orbit_nn_median": float(np.median(nn["max_cosine"])),
           "orbit_copy_fraction": float(copies.mean()),
           "orbit_copies_via_transform": float(transformed[copies].mean()) if copies.any() else 0.0,
           "orbit_nn": nn["max_cosine"], "orbit_ref": ref}
    if heldout is not None:
        val = orbit_max_cosine(as_maps(heldout, "held-out maps"), train, device=device)["max_cosine"]
        out["heldout_orbit_copy_fraction"] = float(np.mean(val > threshold))
        out["heldout_orbit_nn"] = val
    return out


def patch_grid(size: int, p: int, n_locations: int = 16) -> np.ndarray:
    """(n_locations, 2) top-left corners on a fixed sqrt(n) x sqrt(n) grid inside [0, size - p] (no wrap)."""
    k = int(round(np.sqrt(n_locations)))
    if k * k != int(n_locations) or k < 1:
        raise ValueError(f"n_locations must be a perfect square, got {n_locations}")
    if not 1 < int(p) <= int(size):
        raise ValueError(f"patch size {p} must be in (1, {size}]")
    c = np.unique(np.round(np.linspace(0, int(size) - int(p), k)).astype(np.int64))
    if len(c) != k:
        raise ValueError(f"p={p} leaves fewer than {k} distinct grid positions in a {size} map")
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()], axis=1)


def extract_patches(maps: np.ndarray, map_index: np.ndarray, p: int, locations: np.ndarray):
    """Patches of maps[map_index] at every location; returns (patches, map index per patch, location index)."""
    x = as_maps(maps)
    map_index = np.asarray(map_index, dtype=np.int64)
    patches = np.stack([x[i, y:y + p, xx:xx + p] for i in map_index for y, xx in locations])
    return (patches, np.repeat(map_index, len(locations)),
            np.tile(np.arange(len(locations)), len(map_index)))


def patch_orbit_reference(train: np.ndarray, heldout: np.ndarray, p: int, n_maps: int = 64,
                          n_locations: int = 16, quantile: float = 0.95, device: str | None = None) -> dict:
    """Per-N part of patch_orbit_stats: train-train threshold and held-out real patch scores.

    Threshold: quantile of the best patch match of training patches against all training maps,
    the patch's own map excluded (every window, every D4 element). Same n_maps evenly spaced maps
    and the same grid locations as the generated set.
    """
    from simdiff_eval.orbit_nn import patch_orbit_max_cosine, prepare_patch_references

    train, heldout = as_maps(train, "training maps"), as_maps(heldout, "held-out maps")
    locs = patch_grid(train.shape[-1], p, n_locations)
    prepared = prepare_patch_references(train, p, device=device)
    t_idx = np.linspace(0, len(train) - 1, min(len(train), int(n_maps)), dtype=np.int64)
    t_patch, t_map, _ = extract_patches(train, t_idx, p, locs)
    t_nn = patch_orbit_max_cosine(t_patch, None, p, prepared=prepared, exclude_ref_index=t_map, device=device)
    h_idx = np.linspace(0, len(heldout) - 1, min(len(heldout), int(n_maps)), dtype=np.int64)
    h_patch, h_map, _ = extract_patches(heldout, h_idx, p, locs)
    h_nn = patch_orbit_max_cosine(h_patch, None, p, prepared=prepared, device=device)
    return {"p": int(p), "n_maps": int(n_maps), "n_locations": int(n_locations), "quantile": float(quantile),
            "locations": locs, "prepared": prepared, "threshold": float(np.quantile(t_nn["max_cosine"], quantile)),
            "train_nn": t_nn["max_cosine"], "train_map": t_map,
            "heldout_nn": h_nn["max_cosine"], "heldout_map": h_map}


def _median_by_resampled_maps(values: np.ndarray, map_ids: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Median over all patches of each bootstrap draw of maps; draws: (B, n_maps) positions into unique maps."""
    uniq, inv = np.unique(map_ids, return_inverse=True)
    per_map = [values[inv == m] for m in range(len(uniq))]
    return np.array([np.median(np.concatenate([per_map[m] for m in d])) for d in draws])


def patch_orbit_stats(train: np.ndarray, generated: np.ndarray, heldout: np.ndarray | None, p: int,
                      n_maps: int = 64, n_locations: int = 16, quantile: float = 0.95, device: str | None = None,
                      seed: int = 0, n_boot: int = 1000, reference: dict | None = None) -> dict:
    """Patch-mosaic test: best D4 x periodic-window match of p x p patches against the training maps.

    n_maps evenly spaced generated (and held-out) maps, n_locations patches each on a fixed grid.
    excess = median best-match cosine of generated patches - median of held-out real patches, with a
    95% bootstrap CI that resamples maps (generated and held-out independently, seeded).
    Copy fractions use the train-train quantile threshold of patch_orbit_reference.
    Pass ``reference`` (from patch_orbit_reference) to reuse the per-N part across runs.
    """
    from simdiff_eval.orbit_nn import patch_orbit_max_cosine

    if reference is None:
        if heldout is None:
            raise ValueError("patch_orbit_stats needs held-out maps or a precomputed reference")
        reference = patch_orbit_reference(train, heldout, p, n_maps, n_locations, quantile, device)
    if reference["p"] != int(p) or reference["n_locations"] != int(n_locations):
        raise ValueError("reference was computed for a different p or n_locations")
    generated = as_maps(generated, "generated maps")
    g_idx = np.linspace(0, len(generated) - 1, min(len(generated), int(n_maps)), dtype=np.int64)
    g_patch, g_map, g_loc = extract_patches(generated, g_idx, p, reference["locations"])
    nn = patch_orbit_max_cosine(g_patch, None, p, prepared=reference["prepared"], device=device)
    gen, held = nn["max_cosine"], reference["heldout_nn"]
    rng = np.random.default_rng(seed)
    n_g, n_h = len(np.unique(g_map)), len(np.unique(reference["heldout_map"]))
    boot = (_median_by_resampled_maps(gen, g_map, rng.integers(0, n_g, size=(n_boot, n_g)))
            - _median_by_resampled_maps(held, reference["heldout_map"], rng.integers(0, n_h, size=(n_boot, n_h))))
    thr = reference["threshold"]
    return {"nn_median": float(np.median(gen)), "heldout_nn_median": float(np.median(held)),
            "excess": float(np.median(gen) - np.median(held)),
            "excess_ci_low": float(np.quantile(boot, 0.025)), "excess_ci_high": float(np.quantile(boot, 0.975)),
            "threshold": thr, "copy_fraction": float(np.mean(gen > thr)),
            "heldout_copy_fraction": float(np.mean(held > thr)),
            "train_nn_median": float(np.median(reference["train_nn"])),
            "n_generated_patches": int(len(gen)), "n_heldout_patches": int(len(held)),
            "gen_nn": gen, "gen_map": g_map, "gen_location": g_loc, "gen_ref_index": nn["ref_index"],
            "gen_group_element": nn["group_element"], "gen_shift_y": nn["shift_y"], "gen_shift_x": nn["shift_x"],
            "heldout_nn": held, "train_nn": reference["train_nn"], "locations": reference["locations"]}


def plain_nearest_cosine(train: np.ndarray, generated: np.ndarray) -> np.ndarray:
    from simdiff_eval.conditional_unet_diagnostics import nearest_centered_pixel_cosine

    return nearest_centered_pixel_cosine(as_maps(generated), as_maps(train))[0]


def draw_to_draw_cosine(maps: np.ndarray) -> dict:
    from simdiff_eval.conditional_unet_diagnostics import pairwise_centered_cosine_stats

    return pairwise_centered_cosine_stats(as_maps(maps))


# ---------------------------------------------------------------- P(k)
def pk_binning(height: int = 128, n_bins: int = 25) -> dict:
    ky = np.fft.fftfreq(height) * height
    kk = np.sqrt(ky[None, :] ** 2 + ky[:, None] ** 2)
    valid = kk > 0
    edges = np.linspace(kk[valid].min(), kk[valid].max(), n_bins + 1)
    kc = 0.5 * (edges[:-1] + edges[1:])
    bin_of = np.clip(np.digitize(kk, edges) - 1, 0, n_bins - 1)[valid]
    keep = kc <= height // 2
    bands = {"large": kc < 8, "mid": (kc >= 8) & (kc < 24), "small": (kc >= 24) & keep}
    return {"kc": kc, "edges": edges, "valid": valid, "bin_of": bin_of,
            "counts": np.bincount(bin_of, minlength=n_bins), "keep": keep, "bands": bands, "height": height}


def power_spectra(maps: np.ndarray, binning: dict | None = None, batch: int = 512) -> np.ndarray:
    """Per-map mean-subtracted 2-D power in radial bins (notebook pk())."""
    x = as_maps(maps).astype(np.float64)
    b = binning or pk_binning(x.shape[-1])
    h = b["height"]
    nb = len(b["kc"])
    x = x - x.mean(axis=(1, 2), keepdims=True)
    out = np.empty((len(x), nb))
    for s in range(0, len(x), batch):
        f = np.fft.fft2(x[s:s + batch])
        p = ((f * f.conj()).real / (h * h))[:, b["valid"]]
        out[s:s + batch] = np.stack([np.bincount(b["bin_of"], weights=r, minlength=nb) for r in p]) / b["counts"]
    return out


def pk_band_values(ratios: np.ndarray, binning: dict) -> dict:
    """ratios: (groups, bins). Band value = exp(mean over groups of mean log ratio in band)."""
    r = np.atleast_2d(np.asarray(ratios, dtype=np.float64))
    out = {}
    for name, mask in binning["bands"].items():
        lr = np.log(r[:, mask]).mean(axis=1)
        out[f"pk_{name}"] = float(np.exp(lr.mean()))
        out[f"pk_{name}_scatter_pct"] = float(100 * lr.std(ddof=1)) if len(lr) > 1 else float("nan")
    return out


# ---------------------------------------------------------------- one-point
def one_point_moments(maps: np.ndarray) -> dict:
    from scipy import stats

    v = as_maps(maps).ravel().astype(np.float64)
    q01, q50, q99 = np.quantile(v, [0.01, 0.5, 0.99])
    return {"mean": float(v.mean()), "std": float(v.std()), "skew": float(stats.skew(v)),
            "excess_kurtosis": float(stats.kurtosis(v)), "q01": float(q01), "q50": float(q50), "q99": float(q99)}


def one_point_edges(*reference_sets: np.ndarray, bins: int = 120) -> np.ndarray:
    lo = min(float(np.min(r)) for r in reference_sets)
    hi = max(float(np.max(r)) for r in reference_sets)
    return np.linspace(lo, hi, int(bins) + 1)


def one_point_l1(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> dict:
    """PDF L1 on shared bins; mass of either set outside the bins is reported, not dropped silently."""
    va, vb = as_maps(a).ravel(), as_maps(b).ravel()
    ha, _ = np.histogram(va, bins=edges, density=False)
    hb, _ = np.histogram(vb, bins=edges, density=False)
    width = np.diff(edges)
    pa, pb = ha / (len(va) * width), hb / (len(vb) * width)
    outside = lambda v: float(np.mean((v < edges[0]) | (v > edges[-1])))
    return {"l1": float(np.sum(np.abs(pa - pb) * width)), "outside_a": outside(va), "outside_b": outside(vb)}


# ---------------------------------------------------------------- probe space
def tanh_inverse_to_log(model_maps: np.ndarray, center: float, xmax: float, norm_kwargs: dict) -> np.ndarray:
    """Invert log -> center-max -> asymmetric tanh (cosmodiff tanh_norm inverse) back to the log field."""
    alpha = float(norm_kwargs.get("alpha", 1.0))
    beta = float(norm_kwargs.get("beta", 1.0))
    gamma = float(norm_kwargs.get("gamma", 1.0))
    delta = float(norm_kwargs.get("delta", 1.0))
    sigma = float(norm_kwargs.get("sigma", 1.0))
    mu = float(norm_kwargs.get("mu", 0.0))
    y = np.asarray(model_maps, dtype=np.float64) / sigma
    y = np.clip(y, -beta * (1 - 1e-7), alpha * (1 - 1e-7))
    x = np.where(y >= 0, alpha * np.arctanh(y / alpha) / gamma, beta * np.arctanh(y / beta) / delta) + mu
    return x * float(xmax) + float(center)


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(np.ravel(a), np.ravel(b)))


# ---------------------------------------------------------------- conditional recovery
def recovery_summary(points, parameter: str, width: float | None = None) -> dict:
    """Median bias, 68% coverage over held-out cosmologies (conditional_label_sweeps_review cell 12/14)."""
    g = points[points.parameter == parameter]
    bias = g.theta_rec_median - g.theta_in
    covered = (g.theta_rec_q16 <= g.theta_in) & (g.theta_in <= g.theta_rec_q84)
    out = {"median_bias": float(bias.median()), "mean_bias": float(bias.mean()),
           "median_abs_error": float(bias.abs().median()), "covered68": int(covered.sum()),
           "n_cosmologies": int(g.heldout_sim.nunique()), "coverage68": float(covered.mean()),
           "median_width68": float((g.theta_rec_q84 - g.theta_rec_q16).median())}
    if width:
        out["median_bias_over_W"] = out["median_bias"] / width
    return out


# ---------------------------------------------------------------- verdict
def memorized(g_d4: float, orbit_copy_fraction: float | None) -> bool:
    """Ledger unet128_aug_sweep_eval definition: G(d4) < 0.5 OR orbit copy fraction >= 0.5."""
    return bool(g_d4 < 0.5 or (orbit_copy_fraction is not None and np.isfinite(orbit_copy_fraction)
                               and orbit_copy_fraction >= 0.5))
