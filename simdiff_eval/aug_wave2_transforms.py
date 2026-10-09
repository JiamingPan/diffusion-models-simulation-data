"""Augmentation-sweep wave 2 transforms (ledger unet128_aug_wave2); no training side effects.

Factorises simdiff_eval.unet64_sweep_transforms.SymmetryPreserving (random D4
element + random periodic roll per fetch) into its parts:

  ShiftOnly : random periodic roll on both axes, no D4 element
  D4Only    : random D4 element, no roll
  FlipOnly  : identity or left-right flip (flip of the last axis), p = 0.5 each, no roll

Same call signature, private numpy Generator seeded at construction, numpy and
torch branches, applied per fetch to already-normalized maps (cosmodiff
Dataset.__getitem__). D4 element g uses the convention of
simdiff_eval.d4_unet64.d4: rot90 by g % 4 on the last two axes, then flip of the
last axis if g >= 4. The left-right flip of FlipOnly is element 4 in that order.
"""
import numpy as np


def _d4(image, element):
    if isinstance(image, np.ndarray):
        out = np.rot90(image, element % 4, axes=(-2, -1))
        return np.flip(out, axis=-1) if element >= 4 else out
    import torch
    out = torch.rot90(image, element % 4, dims=(-2, -1))
    return out.flip(-1) if element >= 4 else out


def _finish(image):
    if isinstance(image, np.ndarray):
        return image.copy()
    return image.contiguous()


def _square(image):
    if image.shape[-2] != image.shape[-1]:
        raise ValueError('Square maps required')


class ShiftOnly:
    """Random periodic roll (dy, dx), each uniform on 0..H-1; no rotation or flip."""

    def __init__(self, seed=123):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        _square(image)
        shifts = tuple(map(int, self.rng.integers(image.shape[-1], size=2)))
        if isinstance(image, np.ndarray):
            return np.roll(image, shifts, axis=(-2, -1)).copy()
        import torch
        return torch.roll(image, shifts, dims=(-2, -1)).contiguous()

    def __repr__(self):
        return f'ShiftOnly(seed={self.seed})'


class D4Only:
    """Random D4 element, uniform over the 8 elements; no roll."""

    def __init__(self, seed=123):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        _square(image)
        element = int(self.rng.integers(8))
        return _finish(_d4(image, element))

    def __repr__(self):
        return f'D4Only(seed={self.seed})'


class FlipOnly:
    """Identity or left-right flip (D4 element 4), probability 0.5 each; no roll."""

    def __init__(self, seed=123):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        _square(image)
        element = 4 * int(self.rng.integers(2))
        return _finish(_d4(image, element))

    def __repr__(self):
        return f'FlipOnly(seed={self.seed})'


ARMS = {'shift_only': ShiftOnly, 'd4_only': D4Only, 'flip_only': FlipOnly}
