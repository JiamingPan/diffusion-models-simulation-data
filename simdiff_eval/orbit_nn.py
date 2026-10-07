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
                     device: str | None = None, query_batch: int = 4, ref_batch: int = 512) -> dict:
    """Best orbit cosine of each query against all references.

    exclude_same_index: queries and references are the same set; query i never
    matches reference i under any transform (its whole orbit is excluded), as
    needed for train-to-train threshold calibration.
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
            if exclude_same_index:
                for b in range(len(qb)):
                    j = q0 + b - r0
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
