"""Augmentation wave 3 transforms (ledger unet128_aug_wave3, prepared 2026-10-23); no training side effects.

Restricted periodic-roll sets that test the patch-orbit posterior collapse noise sigma_c
(advisor analysis, orbit_posterior_entropy.py SUBSHIFT):

  JitterRoll(max_shift=4) : roll (dy, dx), each uniform on {0, 1, 2, 3}           -> 16 views
  StrideRoll(stride=32)   : roll (dy, dx), each uniform on {0, 32, 64, 96} (H=128)  -> 16 views
  ShiftOnly               : re-exported unchanged from simdiff_eval.aug_wave2_transforms
                            (uniform 0..H-1 on both axes, H*H views)
  D4Only                  : re-exported unchanged from simdiff_eval.aug_wave2_transforms (8 views)
  SymmetryPreserving      : re-exported unchanged from simdiff_eval.unet64_sweep_transforms, the
                            transform of the wave-1 d4shift arm (scripts/run_unet64_three_method.py
                            --arm d4_periodic_shift): random D4 element then uniform roll, 8*H*H views

Same convention as aug_wave2_transforms: private numpy Generator seeded at construction,
one rng.integers(len(shift_set), size=2) draw per call (dy index, dx index), numpy and
torch branches, applied per fetch to already-normalized maps (cosmodiff
ArrayDataset.__getitem__). Roll sign matches np.roll(x, (dy, dx), axis=(-2, -1)), i.e.
output(p) = x(p - s), the convention of the advisor's terms_fft. The views are a set of
rolls, not a group (jitter is not closed under composition); each fetch applies one roll
to the stored map, so there is no composition.
"""
import numpy as np

from simdiff_eval.aug_wave2_transforms import D4Only, ShiftOnly, _square
from simdiff_eval.unet64_sweep_transforms import SymmetryPreserving


class SubsetRoll:
    """Periodic roll by (dy, dx), dy and dx drawn independently and uniformly from a 1-d offset set."""

    def __init__(self, shifts, seed=123):
        shifts = tuple(int(s) for s in shifts)
        if not shifts or len(set(shifts)) != len(shifts) or min(shifts) < 0:
            raise ValueError(f'Offsets must be distinct non-negative integers, got {shifts}')
        self.shifts = shifts
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def shift_set(self, size):
        """Offsets used on each axis for a size x size map."""
        if max(self.shifts) >= size:
            raise ValueError(f'Offset {max(self.shifts)} does not fit a {size}-pixel axis')
        return self.shifts

    def views(self, size):
        """All (dy, dx) pairs this transform can draw, in draw-index order."""
        s = self.shift_set(size)
        return [(a, b) for a in s for b in s]

    def __call__(self, image):
        _square(image)
        offsets = self.shift_set(image.shape[-1])
        iy, ix = map(int, self.rng.integers(len(offsets), size=2))
        shifts = (offsets[iy], offsets[ix])
        if isinstance(image, np.ndarray):
            return np.roll(image, shifts, axis=(-2, -1)).copy()
        import torch
        return torch.roll(image, shifts, dims=(-2, -1)).contiguous()

    def __repr__(self):
        return f'SubsetRoll(shifts={self.shifts}, seed={self.seed})'


class JitterRoll(SubsetRoll):
    """Small-jitter roll: offsets 0..max_shift-1 on each axis (max_shift=4 -> {0,1,2,3}^2, 16 views)."""

    def __init__(self, max_shift=4, seed=123):
        if int(max_shift) < 1:
            raise ValueError('max_shift must be >= 1')
        self.max_shift = int(max_shift)
        super().__init__(range(self.max_shift), seed)

    def __repr__(self):
        return f'JitterRoll(max_shift={self.max_shift}, seed={self.seed})'


class StrideRoll(SubsetRoll):
    """Coarse roll: offsets 0, stride, 2*stride, ... < H on each axis (stride 32, H 128 -> 16 views)."""

    def __init__(self, stride=32, seed=123):
        if int(stride) < 1:
            raise ValueError('stride must be >= 1')
        self.stride = int(stride)
        self.shifts = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def shift_set(self, size):
        if size % self.stride:
            raise ValueError(f'stride {self.stride} does not divide map size {size}')
        return tuple(range(0, size, self.stride))

    def __repr__(self):
        return f'StrideRoll(stride={self.stride}, seed={self.seed})'


def jitter4(seed):
    return JitterRoll(4, seed)


def stride32(seed):
    return StrideRoll(32, seed)


# arm name (runner --arm) -> factory(seed); instances are picklable (cosmodiff may pickle augmentations)
# d4_periodic_shift / d4_only / shift_only are the wave-1 / wave-2 classes themselves (not copies).
ARMS = {'jitter4': jitter4, 'stride32': stride32, 'shift_only': ShiftOnly,
        'd4_periodic_shift': SymmetryPreserving, 'd4_only': D4Only}
VIEWS_128 = {'jitter4': 16, 'stride32': 16, 'shift_only': 128 * 128,
             'd4_periodic_shift': 8 * 128 * 128, 'd4_only': 8}
