from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
from simdiff_eval import run_metrics as rm  # noqa: E402
from simdiff_eval.orbit_nn import orbit_max_cosine, patch_orbit_max_cosine  # noqa: E402


def smooth_fields(n, size=32, seed=0):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(size)
    kk = np.hypot(k[:, None], k[None, :])
    amp = np.where(kk > 0, kk ** -1.5, 0.0)
    noise = rng.normal(size=(n, size, size)) + 1j * rng.normal(size=(n, size, size))
    return np.fft.ifft2(noise * amp).real.astype(np.float32)


def brute_force_patch(q, refs):
    """Max over D4 of q, every periodic window of every reference, of the centred cosine."""
    p = q.shape[0]
    def unit(x):
        x = x - x.mean()
        return x / np.linalg.norm(x)
    best = -np.inf
    for g in range(8):
        y = np.rot90(q, g % 4)
        y = unit(np.flip(y, -1) if g >= 4 else y)
        for r in refs:
            for sy in range(r.shape[0]):
                for sx in range(r.shape[1]):
                    win = np.roll(r, (-sy, -sx), (0, 1))[:p, :p]
                    best = max(best, float(np.sum(y * unit(win))))
    return best


def test_full_size_patch_equals_map_orbit_search():
    rng = np.random.default_rng(0)
    refs = rng.normal(size=(6, 16, 16)).astype(np.float32)
    queries = rng.normal(size=(5, 16, 16)).astype(np.float32)
    full = orbit_max_cosine(queries, refs, device="cpu")
    patch = patch_orbit_max_cosine(queries, refs, 16, device="cpu", query_batch=2, ref_batch=4)
    np.testing.assert_allclose(patch["max_cosine"], full["max_cosine"], atol=1e-5)
    np.testing.assert_array_equal(patch["ref_index"], full["ref_index"])


def test_rolled_rotated_training_patch_in_random_map_scores_one():
    train = smooth_fields(8, size=32, seed=1)
    p = 8
    # Training map 3, rolled so its patch wraps the periodic edge, then rotated (g = 6).
    src = np.roll(train[3], (-28, -27), (0, 1))[:p, :p]
    patch = rm.d4_element(src[None], 6)[0]
    host = np.random.default_rng(2).normal(size=(1, 32, 32)).astype(np.float32)
    host[0, 10:10 + p, 5:5 + p] = patch
    q, _, _ = rm.extract_patches(host, np.array([0]), p, np.array([[10, 5], [0, 20]]))
    out = patch_orbit_max_cosine(q, train, p, device="cpu")
    assert out["max_cosine"][0] == pytest.approx(1.0, abs=1e-5)
    assert out["ref_index"][0] == 3
    assert (out["shift_y"][0], out["shift_x"][0]) == (28, 27)
    assert out["max_cosine"][1] < 0.99  # pure noise patch does not match


def test_fft_matches_brute_force_on_small_maps():
    rng = np.random.default_rng(3)
    refs = rng.normal(size=(3, 16, 16)).astype(np.float32)
    queries = rng.normal(size=(4, 4, 4)).astype(np.float32)
    out = patch_orbit_max_cosine(queries, refs, 4, device="cpu", query_batch=3, ref_batch=2)
    expected = np.array([brute_force_patch(q, refs) for q in queries])
    np.testing.assert_allclose(out["max_cosine"], expected, atol=1e-5)


def test_exclude_ref_index_and_flat_windows():
    rng = np.random.default_rng(4)
    refs = rng.normal(size=(4, 16, 16)).astype(np.float32)
    q, qmap, _ = rm.extract_patches(refs, np.arange(4), 4, np.array([[2, 3]]))
    out = patch_orbit_max_cosine(q, refs, 4, exclude_ref_index=qmap, device="cpu")
    assert np.all(out["ref_index"] != qmap) and np.all(out["max_cosine"] < 0.999)
    flat = np.zeros((1, 16, 16), dtype=np.float32)
    flat[0, :8] = 1.0  # windows inside a constant half have zero variance
    res = patch_orbit_max_cosine(q[:1], flat, 4, device="cpu")
    assert np.isfinite(res["max_cosine"][0]) and res["ref_index"][0] == 0


def test_patch_orbit_stats_finite_and_zero_excess_when_generated_is_heldout():
    train = smooth_fields(12, size=32, seed=5)
    heldout = smooth_fields(10, size=32, seed=6)
    st = rm.patch_orbit_stats(train, heldout, heldout, 8, n_maps=6, n_locations=4, device="cpu", n_boot=200)
    for k in ("nn_median", "heldout_nn_median", "excess", "excess_ci_low", "excess_ci_high", "threshold",
              "copy_fraction", "heldout_copy_fraction"):
        assert np.isfinite(st[k]), k
    assert st["excess"] == pytest.approx(0.0, abs=1e-7)
    assert st["excess_ci_low"] <= 0.0 <= st["excess_ci_high"]
    assert st["copy_fraction"] == st["heldout_copy_fraction"]
    assert len(st["gen_nn"]) == 24
    copies = rm.patch_orbit_stats(train, train, heldout, 8, n_maps=6, n_locations=4, device="cpu", n_boot=200)
    assert copies["nn_median"] == pytest.approx(1.0, abs=1e-5) and copies["excess"] > 0
    assert copies["copy_fraction"] == 1.0


def test_patch_grid_is_fixed_and_inside_the_map():
    locs = rm.patch_grid(128, 16, 16)
    assert locs.shape == (16, 2) and locs.min() == 0 and locs.max() == 112
    assert len(rm.patch_grid(128, 128, 1)) == 1
    with pytest.raises(ValueError):
        rm.patch_grid(128, 16, 15)


def test_evaluate_run_parses_patch_scales():
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_run

    assert evaluate_run.parse_patch_scales("") == []
    assert evaluate_run.parse_patch_scales("8,16,32,64") == [8, 16, 32, 64]
    with pytest.raises(SystemExit):
        evaluate_run.parse_patch_scales("8,8")
