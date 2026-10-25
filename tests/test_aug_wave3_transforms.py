"""Tests for simdiff_eval.aug_wave3_transforms (ledger unet128_aug_wave3)."""
import pickle

import numpy as np
import pytest

from simdiff_eval import aug_wave2_transforms as w2
from simdiff_eval import unet64_sweep_transforms as w1
from simdiff_eval.aug_wave3_transforms import (ARMS, VIEWS_128, D4Only, JitterRoll, ShiftOnly, StrideRoll, SubsetRoll,
                                               SymmetryPreserving)

H = 128
DRAWS = 8000
CHI2_001_DF15 = 37.70   # chi-square critical value, p = 0.001, df = 15

EXPECTED_SETS = {'jitter4': (0, 1, 2, 3), 'stride32': (0, 32, 64, 96)}


def ramp(size=H):
    # every pixel value distinct, so the roll that produced an output is identified by where 0 lands
    return np.arange(size * size, dtype=np.float32).reshape(1, size, size)


def roll_of(x, y):
    """(dy, dx) with y == np.roll(x, (dy, dx)), checked exactly; x must be ramp()."""
    dy, dx = np.unravel_index(int(np.argmin(y[0])), y.shape[-2:])
    assert np.array_equal(y, np.roll(x, (int(dy), int(dx)), axis=(-2, -1))), 'output is not a periodic roll'
    return int(dy), int(dx)


def chi2(counts):
    counts = np.asarray(counts, float)
    expected = counts.sum() / len(counts)
    return float(((counts - expected) ** 2 / expected).sum())


@pytest.mark.parametrize('arm', ['jitter4', 'stride32'])
def test_shift_set_is_exactly_as_specified_and_uniform(arm):
    x, t = ramp(), ARMS[arm](seed=5)
    offs = EXPECTED_SETS[arm]
    assert t.shift_set(H) == offs
    assert t.views(H) == [(a, b) for a in offs for b in offs] and VIEWS_128[arm] == 16
    counts = {}
    for _ in range(DRAWS):
        s = roll_of(x, t(x))
        counts[s] = counts.get(s, 0) + 1
    assert set(counts) == {(a, b) for a in offs for b in offs}, sorted(counts)
    assert chi2(list(counts.values())) < CHI2_001_DF15


@pytest.mark.parametrize('arm, cls, views', [('shift_only', w2.ShiftOnly, H * H), ('d4_only', w2.D4Only, 8),
                                             ('d4_periodic_shift', w1.SymmetryPreserving, 8 * H * H)])
def test_reused_arms_are_the_wave1_wave2_classes(arm, cls, views):
    # reuse, not reimplementation: the registry holds the earlier classes themselves
    assert ARMS[arm] is cls and VIEWS_128[arm] == views
    assert ShiftOnly is w2.ShiftOnly and D4Only is w2.D4Only and SymmetryPreserving is w1.SymmetryPreserving
    a, b = ARMS[arm](1123), cls(1123)
    x = ramp()
    for _ in range(50):
        np.testing.assert_array_equal(a(x), b(x))


def test_d4shift_matches_wave1_runner_construction():
    # scripts/run_unet64_three_method.py: dataset.augmentations = SymmetryPreserving(args.seed + 1000)
    a, b = ARMS['d4_periodic_shift'](123 + 1000), w1.SymmetryPreserving(1123)
    x = np.random.default_rng(0).normal(size=(1, H, H)).astype('float32')
    for _ in range(50):
        np.testing.assert_array_equal(a(x), b(x))


def test_d4_arms_reach_their_views():
    x = ramp(8)
    t = ARMS['d4_only'](0)
    seen_d4 = {t(x).tobytes() for _ in range(400)}
    assert len(seen_d4) == 8
    t = ARMS['d4_periodic_shift'](0)
    seen = {t(x).tobytes() for _ in range(6000)}
    assert len(seen) == 8 * 8 * 8   # D4 x 8x8 rolls on an 8x8 map with distinct pixels


def test_rolls_are_periodic():
    # pixels leaving one edge re-enter on the opposite edge; no padding, no interpolation
    x = ramp(8)
    t = SubsetRoll((3,), seed=0)
    y = t(x)
    np.testing.assert_array_equal(y[0, 3:, 3:], x[0, :5, :5])
    np.testing.assert_array_equal(y[0, :3, :3], x[0, 5:, 5:])
    np.testing.assert_array_equal(np.roll(y, (-3, -3), axis=(-2, -1)), x)
    # stride equal to the map size is the identity only; offsets wrap modulo H
    assert StrideRoll(8).shift_set(8) == (0,)
    np.testing.assert_array_equal(StrideRoll(8, seed=1)(x), x)


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_deterministic_under_seed_and_seed_dependent(arm):
    x = ramp()
    a, b, c = ARMS[arm](123), ARMS[arm](123), ARMS[arm](124)
    ya = [a(x) for _ in range(50)]
    yb = [b(x) for _ in range(50)]
    yc = [c(x) for _ in range(50)]
    assert all(np.array_equal(p, q) for p, q in zip(ya, yb))
    assert not all(np.array_equal(p, q) for p, q in zip(ya, yc))


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_pixel_pdf_and_power_preserved(arm):
    x = np.random.default_rng(2).normal(size=(1, H, H)).astype('float32')
    t = ARMS[arm](7)
    pk = lambda a: (abs(np.fft.fft2(a[0])) ** 2)
    for _ in range(20):
        y = t(x)
        assert y.shape == x.shape and y.dtype == x.dtype and y.flags['C_CONTIGUOUS']
        assert np.array_equal(np.sort(x.ravel()), np.sort(y.ravel()))   # identical one-point PDF
        if arm.startswith('d4'):
            # rotations / flips permute Fourier modes: the multiset of |FFT|^2 is unchanged
            np.testing.assert_allclose(np.sort(pk(x).ravel()), np.sort(pk(y).ravel()), rtol=1e-4, atol=1e-2)
        else:
            np.testing.assert_allclose(pk(x), pk(y), rtol=1e-4, atol=1e-2)  # |FFT|^2 per mode unchanged by a roll


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_torch_branch_matches_numpy_branch(arm):
    torch = pytest.importorskip('torch')
    x = ramp()
    tn, tt = ARMS[arm](9), ARMS[arm](9)
    for _ in range(64):
        yn = tn(x)
        yt = tt(torch.from_numpy(x.copy()))
        assert isinstance(yt, torch.Tensor) and yt.is_contiguous()
        np.testing.assert_array_equal(yt.numpy(), yn)


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_pickle_round_trip_keeps_rng_state(arm):
    x, t = ramp(), ARMS[arm](3)
    for _ in range(5):
        t(x)
    u = pickle.loads(pickle.dumps(t))
    if arm != 'd4_periodic_shift':   # wave-1 SymmetryPreserving has no custom __repr__
        assert repr(u) == repr(t)
    for _ in range(20):
        np.testing.assert_array_equal(t(x), u(x))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        JitterRoll(4)(np.zeros((1, 8, 6)))          # non-square
    with pytest.raises(ValueError):
        StrideRoll(48)(np.zeros((1, 128, 128)))     # stride does not divide H
    with pytest.raises(ValueError):
        JitterRoll(9)(np.zeros((1, 8, 8)))          # offset larger than the axis
    with pytest.raises(ValueError):
        SubsetRoll((1, 1))
    with pytest.raises(ValueError):
        JitterRoll(0)


def test_repr_and_registry():
    assert repr(ARMS['jitter4'](1123)) == 'JitterRoll(max_shift=4, seed=1123)'
    assert repr(ARMS['stride32'](1123)) == 'StrideRoll(stride=32, seed=1123)'
    assert set(ARMS) == {'jitter4', 'stride32', 'shift_only', 'd4_periodic_shift', 'd4_only'}
