"""Nearest-training similarity that allows D4 transforms and periodic shifts.

A plain nearest-neighbour cosine misses a generated map that is a rotated,
flipped or rolled copy of a training map, so augmentation arms could look
novel while copying. Here the similarity of a query q to a reference r is

    max over g in D4 and all circular shifts s of cos(g(q) rolled by s, r)

with each map mean-subtracted and unit-normalised (the same centred cosine as
``pairwise_centered_cosine_stats``). Rolling preserves mean and norm, so every
circular cross-correlation value is a cosine and one FFT per (g, q, r) covers
all 128 x 128 shifts.
"""
from __future__ import annotations

import numpy as np


def _unit_centred(maps, torch):
    x = maps.reshape(len(maps), *maps.shape[-2:]).to(torch.float32)
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    norms = x.flatten(1).norm(dim=1)
    if torch.any(norms == 0):
        raise ValueError("constant map has no centred cosine")
    return x / norms[:, None, None]


def _d4_stack(x, torch):
    """(B, H, W) -> (8, B, H, W) in the element order of simdiff_eval.d4_unet64.d4."""
    out = []
    for g in range(8):
        y = torch.rot90(x, g % 4, dims=(-2, -1))
        out.append(y.flip(-1) if g >= 4 else y)
    return torch.stack(out)


def orbit_max_cosine(queries: np.ndarray, references: np.ndarray, *, exclude_same_index: bool = False,
                     exclude_ref_index: np.ndarray | None = None,
                     device: str | None = None, query_batch: int = 4, ref_batch: int = 512) -> dict:
    """Best orbit cosine of each query against all references.

    exclude_same_index: queries and references are the same set; query i never
    matches reference i under any transform (its whole orbit is excluded), as
    needed for train-to-train threshold calibration.
    exclude_ref_index: per-query reference index whose whole orbit is excluded
    (queries a subset of references); -1 excludes nothing.
    Returns arrays: max_cosine, ref_index, group_element, shift_y, shift_x.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q = _unit_centred(torch.as_tensor(np.asarray(queries)), torch)
    r = _unit_centred(torch.as_tensor(np.asarray(references)), torch)
    if q.shape[-2:] != r.shape[-2:] or q.shape[-1] != q.shape[-2]:
        raise ValueError("square maps of equal size required")
    if exclude_same_index and len(q) != len(r):
        raise ValueError("exclude_same_index needs queries == references")
    if exclude_same_index and exclude_ref_index is not None:
        raise ValueError("use exclude_same_index or exclude_ref_index, not both")
    if exclude_same_index:
        exclude_ref_index = np.arange(len(q))
    if exclude_ref_index is not None:
        exclude_ref_index = np.asarray(exclude_ref_index, dtype=np.int64).reshape(-1)
        if len(exclude_ref_index) != len(q):
            raise ValueError("exclude_ref_index needs one entry per query")
    h, w = q.shape[-2:]
    n_q = len(q)
    best = torch.full((n_q,), -np.inf)
    best_ref = torch.full((n_q,), -1, dtype=torch.long)
    best_g = torch.full((n_q,), -1, dtype=torch.long)
    best_shift = torch.full((n_q,), -1, dtype=torch.long)
    for r0 in range(0, len(r), ref_batch):
        r_f = torch.fft.rfft2(r[r0:r0 + ref_batch].to(device))           # (R, H, W/2+1)
        n_r = len(r_f)
        for q0 in range(0, n_q, query_batch):
            qb = q[q0:q0 + query_batch].to(device)
            q_f = torch.fft.rfft2(_d4_stack(qb, torch))                    # (8, B, H, W/2+1)
            # corr[g, b, j, s] = sum_x g(q_b)(x + s) r_j(x)
            corr = torch.fft.irfft2(q_f[:, :, None] * r_f.conj()[None, None], s=(h, w))
            corr = corr.reshape(8, len(qb), n_r, h * w)
            if exclude_ref_index is not None:
                for b in range(len(qb)):
                    j = int(exclude_ref_index[q0 + b]) - r0
                    if 0 <= j < n_r:
                        corr[:, b, j] = -np.inf
            vals, flat = corr.permute(1, 0, 2, 3).reshape(len(qb), -1).max(dim=1)
            vals, flat = vals.cpu(), flat.cpu()
            g, rem = flat // (n_r * h * w), flat % (n_r * h * w)
            j, s = rem // (h * w), rem % (h * w)
            better = vals > best[q0:q0 + len(qb)]
            sl = slice(q0, q0 + len(qb))
            best[sl] = torch.where(better, vals, best[sl])
            best_ref[sl] = torch.where(better, j + r0, best_ref[sl])
            best_g[sl] = torch.where(better, g, best_g[sl])
            best_shift[sl] = torch.where(better, s, best_shift[sl])
    return {"max_cosine": best.clamp(-1, 1).numpy(), "ref_index": best_ref.numpy(),
            "group_element": best_g.numpy(), "shift_y": (best_shift // w).numpy(),
            "shift_x": (best_shift % w).numpy()}


# ---------------------------------------------------------------- patch search
# Patch version (Kamb & Ganguli 2412.20292 style): a p x p query patch against every
# periodic p x p window of every reference map, under the 8 D4 elements. The query is
# centred and unit-normalised, so sum_u g(q)(u) W(u) equals sum_u g(q)(u) (W(u) - mean W)
# and the window mean drops out of the numerator; the denominator is the window's
# centred norm sqrt(S2 - S1^2 / p^2) from periodic box sums of T and T^2.

def prepare_patch_references(references: np.ndarray, p: int, *, device: str | None = None,
                             min_rel_var: float = 1e-6) -> dict:
    """Reference spectra and inverse window norms for patch size p (computed once per reference set).

    Window at offset s = (sy, sx) covers T[(sy + u) mod H, (sx + v) mod W], 0 <= u, v < p.
    Windows whose centred sum of squares is below min_rel_var * p^2 * (mean pixel variance of
    the reference set) get inverse norm 0 and are marked invalid (score -inf).
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    r = torch.as_tensor(np.asarray(references))
    r = r.reshape(len(r), *r.shape[-2:]).to(torch.float64)
    h, w = r.shape[-2:]
    p = int(p)
    if h != w or not 1 < p <= h:
        raise ValueError(f"square references and 1 < p <= {h} required, got {tuple(r.shape)} and p={p}")
    box = torch.zeros((h, w), dtype=torch.float64)
    box[:p, :p] = 1.0
    box_f = torch.fft.rfft2(box.to(device)).conj()
    var_ref = float(r.var(dim=(-2, -1)).mean())
    inv_norm, valid, spectra = [], [], []
    for s in range(0, len(r), 256):
        t = r[s:s + 256].to(device)
        s1 = torch.fft.irfft2(box_f * torch.fft.rfft2(t), s=(h, w))
        s2 = torch.fft.irfft2(box_f * torch.fft.rfft2(t * t), s=(h, w))
        ss = s2 - s1 * s1 / (p * p)
        ok = ss > min_rel_var * p * p * max(var_ref, 1e-30)
        inv_norm.append(torch.where(ok, ss.clamp_min(1e-300).rsqrt(), torch.zeros_like(ss)).to(torch.float32).cpu())
        valid.append(ok.cpu())
        spectra.append(torch.fft.rfft2(t.to(torch.float32)).cpu())
    return {"p": p, "shape": (h, w), "n_ref": len(r), "spectra": torch.cat(spectra), "inv_norm": torch.cat(inv_norm),
            "valid": torch.cat(valid), "device": device}


def patch_orbit_max_cosine(queries: np.ndarray, references: np.ndarray | None, p: int, *,
                           prepared: dict | None = None, exclude_ref_index: np.ndarray | None = None,
                           device: str | None = None, query_batch: int = 4, ref_batch: int = 64) -> dict:
    """Best D4 x periodic-window centred cosine of each p x p query patch against all references.

    queries: (Q, p, p) patches. references: (R, H, W) maps, or None with ``prepared`` from
    prepare_patch_references (reused across query sets). exclude_ref_index: per-query reference
    index whose every window and D4 element is excluded (-1 excludes nothing).
    Returns arrays: max_cosine, ref_index, group_element, shift_y, shift_x (window top-left in the
    reference; group element g is applied to the query, same order as _d4_stack).
    """
    import torch

    p = int(p)
    if prepared is None:
        prepared = prepare_patch_references(references, p, device=device)
    elif prepared["p"] != p:
        raise ValueError(f"prepared references are for p={prepared['p']}, not {p}")
    device = device or prepared["device"]
    h, w = prepared["shape"]
    q = _unit_centred(torch.as_tensor(np.asarray(queries)), torch)
    if q.shape[-2:] != (p, p):
        raise ValueError(f"queries must be {p} x {p} patches, got {tuple(q.shape[-2:])}")
    n_q, n_ref = len(q), prepared["n_ref"]
    if exclude_ref_index is not None:
        exclude_ref_index = np.asarray(exclude_ref_index, dtype=np.int64).reshape(-1)
        if len(exclude_ref_index) != n_q:
            raise ValueError("exclude_ref_index needs one entry per query")
    best = torch.full((n_q,), -np.inf)
    best_ref = torch.full((n_q,), -1, dtype=torch.long)
    best_g = torch.full((n_q,), -1, dtype=torch.long)
    best_shift = torch.full((n_q,), -1, dtype=torch.long)
    for r0 in range(0, n_ref, ref_batch):
        t_f = prepared["spectra"][r0:r0 + ref_batch].to(device)            # (R, H, W/2+1)
        inv = prepared["inv_norm"][r0:r0 + ref_batch].to(device)          # (R, H, W)
        invalid = ~prepared["valid"][r0:r0 + ref_batch].to(device)
        n_r = len(t_f)
        for q0 in range(0, n_q, query_batch):
            qb = q[q0:q0 + query_batch]
            pad = torch.zeros((8, len(qb), h, w), dtype=torch.float32)
            pad[..., :p, :p] = _d4_stack(qb, torch)
            q_f = torch.fft.rfft2(pad.to(device)).conj()                     # (8, B, H, W/2+1)
            # corr[g, b, j, s] = sum_u g(q_b)(u) T_j(u + s)
            corr = torch.fft.irfft2(q_f[:, :, None] * t_f[None, None], s=(h, w))
            corr.mul_(inv).masked_fill_(invalid, -np.inf)
            corr = corr.reshape(8, len(qb), n_r, h * w)
            if exclude_ref_index is not None:
                for b in range(len(qb)):
                    j = int(exclude_ref_index[q0 + b]) - r0
                    if 0 <= j < n_r:
                        corr[:, b, j] = -np.inf
            vals, flat = corr.permute(1, 0, 2, 3).reshape(len(qb), -1).max(dim=1)
            vals, flat = vals.cpu(), flat.cpu()
            g, rem = flat // (n_r * h * w), flat % (n_r * h * w)
            j, s = rem // (h * w), rem % (h * w)
            sl = slice(q0, q0 + len(qb))
            better = vals > best[sl]
            best[sl] = torch.where(better, vals, best[sl])
            best_ref[sl] = torch.where(better, j + r0, best_ref[sl])
            best_g[sl] = torch.where(better, g, best_g[sl])
            best_shift[sl] = torch.where(better, s, best_shift[sl])
    return {"max_cosine": best.clamp(-1, 1).numpy(), "ref_index": best_ref.numpy(),
            "group_element": best_g.numpy(), "shift_y": (best_shift // w).numpy(),
            "shift_x": (best_shift % w).numpy()}
