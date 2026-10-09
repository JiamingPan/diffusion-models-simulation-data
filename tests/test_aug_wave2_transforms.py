"""Tests for simdiff_eval.aug_wave2_transforms (ledger unet128_aug_wave2)."""
import pickle

import numpy as np
import pytest

from simdiff_eval.aug_wave2_transforms import ARMS, D4Only, FlipOnly, ShiftOnly

H = 8
DRAWS = 10_000
# chi-square critical values at p = 0.001 (df = 1, 7, 63)
CHI2_001 = {1: 10.83, 7: 24.32, 63: 103.4}


def ref_d4(x, g):
    """Element g written out from the d4_unet64.d4 convention: rot90 by g % 4, then flip last axis if g >= 4."""
    y = np.rot90(x, g % 4, axes=(-2, -1))
    return np.flip(y, -1) if g >= 4 else y


def base_map():
    # generic map: all 8 D4 images and all H*H rolls are distinct
    return np.random.default_rng(11).normal(size=(1, H, H))


def shift_of(x, y):
    hits = [(dy, dx) for dy in range(H) for dx in range(H) if np.array_equal(np.roll(x, (dy, dx), (-2, -1)), y)]
    assert len(hits) <= 1
    return hits[0] if hits else None


def element_of(x, y):
    hits = [g for g in range(8) if np.array_equal(ref_d4(x, g), y)]
    assert len(hits) <= 1
    return hits[0] if hits else None


def chi2(counts):
    counts = np.asarray(counts, float)
    expected = counts.sum() / len(counts)
    return float(((counts - expected) ** 2 / expected).sum())


def test_generic_map_has_distinct_orbit():
    x = base_map()
    assert len({ref_d4(x, g).tobytes() for g in range(8)}) == 8
    assert len({np.roll(x, (a, b), (-2, -1)).tobytes() for a in range(H) for b in range(H)}) == H * H


def test_shift_only_outputs_rolls_uniformly():
    x, t = base_map(), ShiftOnly(seed=5)
    counts = np.zeros((H, H), int)
    for _ in range(DRAWS):
        s = shift_of(x, t(x))
        assert s is not None, 'ShiftOnly output is not a periodic roll of the input'
        counts[s] += 1
    assert chi2(counts.ravel()) < CHI2_001[63]
    assert counts.min() > 0


def test_d4_only_outputs_d4_elements_uniformly_without_roll():
    x, t = base_map(), D4Only(seed=5)
    counts = np.zeros(8, int)
    for _ in range(DRAWS):
        g = element_of(x, t(x))
        assert g is not None, 'D4Only output is not a D4 image of the input (a roll would fail here)'
        counts[g] += 1
    assert chi2(counts) < CHI2_001[7]
    assert counts.min() > 0


def test_flip_only_outputs_identity_or_lr_flip():
    x, t = base_map(), FlipOnly(seed=5)
    counts = np.zeros(2, int)
    for _ in range(DRAWS):
        y = t(x)
        if np.array_equal(y, x):
            counts[0] += 1
        elif np.array_equal(y, x[..., ::-1]):
            counts[1] += 1
        else:
            raise AssertionError('FlipOnly output is neither the input nor its left-right flip')
    assert element_of(x, x[..., ::-1]) == 4
    assert chi2(counts) < CHI2_001[1]


@pytest.mark.parametrize('cls', [ShiftOnly, D4Only, FlipOnly])
def test_deterministic_under_seed_and_seed_dependent(cls):
    x = base_map()
    a, b, c = cls(seed=123), cls(seed=123), cls(seed=124)
    ya = [a(x) for _ in range(50)]
    yb = [b(x) for _ in range(50)]
    yc = [c(x) for _ in range(50)]
    assert all(np.array_equal(p, q) for p, q in zip(ya, yb))
    assert not all(np.array_equal(p, q) for p, q in zip(ya, yc))


@pytest.mark.parametrize('cls', [ShiftOnly, D4Only, FlipOnly])
def test_preserves_shape_dtype_pixels_and_power(cls):
    x = np.random.default_rng(2).normal(size=(1, 32, 32)).astype('float32')
    t = cls(seed=7)
    pk = lambda a: np.sort((abs(np.fft.fft2(a[0])) ** 2).ravel())
    for _ in range(20):
        y = t(x)
        assert y.shape == x.shape and y.dtype == x.dtype and y.flags['C_CONTIGUOUS']
        assert np.array_equal(np.sort(x.ravel()), np.sort(y.ravel()))
        np.testing.assert_allclose(pk(x), pk(y), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('cls', [ShiftOnly, D4Only, FlipOnly])
def test_non_square_rejected(cls):
    with pytest.raises(ValueError):
        cls()(np.zeros((1, 8, 6)))


@pytest.mark.parametrize('cls', [ShiftOnly, D4Only, FlipOnly])
def test_torch_branch_matches_numpy_branch(cls):
    torch = pytest.importorskip('torch')
    x = base_map().astype('float32')
    tn, tt = cls(seed=9), cls(seed=9)
    for _ in range(64):
        yn = tn(x)
        yt = tt(torch.from_numpy(x.copy()))
        assert isinstance(yt, torch.Tensor) and yt.is_contiguous()
        np.testing.assert_array_equal(yt.numpy(), yn)


def test_d4_convention_matches_d4_unet64():
    torch = pytest.importorskip('torch')
    pytest.importorskip('diffusers')
    from simdiff_eval.d4_unet64 import d4
    from simdiff_eval.aug_wave2_transforms import _d4

    x = torch.from_numpy(base_map().astype('float32'))
    for g in range(8):
        np.testing.assert_array_equal(_d4(x, g).numpy(), d4(x, g).numpy())
        np.testing.assert_array_equal(_d4(x.numpy(), g), d4(x, g).numpy())


@pytest.mark.parametrize('cls', [ShiftOnly, D4Only, FlipOnly])
def test_pickle_round_trip_keeps_rng_state(cls):
    # cosmodiff pickles dataset.augmentations into every checkpoint (augmentations.pkl)
    x, t = base_map(), cls(seed=3)
    for _ in range(5):
        t(x)
    u = pickle.loads(pickle.dumps(t))
    assert repr(u) == repr(t)
    for _ in range(20):
        np.testing.assert_array_equal(t(x), u(x))


def test_arm_registry():
    assert ARMS == {'shift_only': ShiftOnly, 'd4_only': D4Only, 'flip_only': FlipOnly}
