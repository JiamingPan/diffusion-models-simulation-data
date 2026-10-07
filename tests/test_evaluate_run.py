from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from simdiff_eval import run_metrics as rm  # noqa: E402


def smooth_fields(n, size=32, seed=0):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(size)
    kk = np.hypot(k[:, None], k[None, :])
    amp = np.where(kk > 0, kk ** -1.5, 0.0)
    noise = rng.normal(size=(n, size, size)) + 1j * rng.normal(size=(n, size, size))
    return np.fft.ifft2(noise * amp).real.astype(np.float32)


def test_orbit_search_finds_rotated_rolled_copy_that_plain_cosine_misses():
    pytest.importorskip("torch")
    from simdiff_eval.orbit_nn import orbit_max_cosine

    train = smooth_fields(12)
    copy = np.roll(rm.d4_element(train[5:6], 5), (7, -3), axis=(-2, -1))
    orbit = orbit_max_cosine(copy, train, device="cpu")
    assert orbit["max_cosine"][0] == pytest.approx(1.0, abs=1e-5)
    assert orbit["ref_index"][0] == 5
    assert rm.plain_nearest_cosine(train, copy)[0] < 0.9


def test_orbit_exclude_ref_index_matches_exclude_same_index():
    pytest.importorskip("torch")
    from simdiff_eval.orbit_nn import orbit_max_cosine

    train = smooth_fields(10, seed=1)
    full = orbit_max_cosine(train, train, exclude_same_index=True, device="cpu")["max_cosine"]
    idx = np.array([1, 4, 8])
    sub = orbit_max_cosine(train[idx], train, exclude_ref_index=idx, device="cpu")["max_cosine"]
    np.testing.assert_allclose(sub, full[idx], atol=1e-6)
    assert np.all(sub < 0.999)


def test_pca_novelty_d4_catches_rotated_copies():
    pytest.importorskip("sklearn")
    train = smooth_fields(60, seed=2)
    space = rm.PCASpace(train, n_components=16)
    exact = rm.pca_novelty(space, train, train[:20], orbit="none")
    rotated = rm.d4_element(train[:20], 1)
    plain = rm.pca_novelty(space, train, rotated, orbit="none")
    orbit = rm.pca_novelty(space, train, rotated, orbit="d4")
    assert exact["G"] == 0.0
    assert plain["G"] > 0.5
    assert orbit["G"] == 0.0


def test_pk_binning_matches_notebook_bands():
    b = rm.pk_binning(128)
    assert len(b["kc"]) == 25
    assert b["bands"]["large"].sum() + b["bands"]["mid"].sum() + b["bands"]["small"].sum() == b["keep"].sum()
    flat = rm.pk_band_values(np.ones((3, 25)), b)
    assert flat["pk_large"] == pytest.approx(1.0) and flat["pk_small"] == pytest.approx(1.0)
    doubled = rm.pk_band_values(2 * np.ones((1, 25)), b)
    assert doubled["pk_mid"] == pytest.approx(2.0)


def test_white_noise_power_is_flat():
    rng = np.random.default_rng(3)
    p = rm.power_spectra(rng.normal(size=(64, 128, 128)).astype(np.float32)).mean(0)
    assert np.max(np.abs(p / p.mean() - 1)) < 0.05


def test_tanh_inverse_round_trips_the_training_normalization():
    from simdiff_eval.io import _normalize_reference_slices

    cfg = {"transform": ["log"], "normalization": "tanh",
           "norm_kwargs": {"alpha": 0.8, "beta": 10.0, "gamma": 1.0, "delta": 1.0, "sigma": 1.5}}
    raw = np.exp(np.random.default_rng(4).normal(8.0, 1.0, size=(3, 16, 16))).astype(np.float32)
    model = _normalize_reference_slices(raw, cfg, 7.9, 22.4)[:, 0]
    back = rm.tanh_inverse_to_log(model, 7.9, 22.4, cfg["norm_kwargs"])
    np.testing.assert_allclose(back, np.log(raw), atol=2e-3)


def test_one_point_l1_and_out_of_range_mass():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 16, 16)).astype(np.float32)
    edges = rm.one_point_edges(a)
    assert rm.one_point_l1(a, a, edges)["l1"] == pytest.approx(0.0)
    shifted = rm.one_point_l1(a + 100, a, edges)
    assert shifted["outside_a"] == pytest.approx(1.0)


def test_memorized_rule_is_the_ledger_or_rule():
    assert rm.memorized(0.2, 0.0)
    assert rm.memorized(0.9, 0.6)
    assert not rm.memorized(0.9, 0.1)
    assert not rm.memorized(0.9, float("nan"))
    assert not rm.memorized(0.9, None)


def test_recovery_summary_counts_coverage_per_cosmology():
    pd = pytest.importorskip("pandas")
    pts = pd.DataFrame({"parameter": ["Omega_m"] * 4, "heldout_sim": [1, 2, 3, 4],
                        "theta_in": [0.2, 0.3, 0.4, 0.5], "theta_rec_median": [0.21, 0.25, 0.4, 0.3],
                        "theta_rec_q16": [0.15, 0.24, 0.35, 0.25], "theta_rec_q84": [0.25, 0.28, 0.45, 0.35]})
    s = rm.recovery_summary(pts, "Omega_m", width=0.1)
    assert s["covered68"] == 2 and s["n_cosmologies"] == 4
    assert s["median_bias"] == pytest.approx(np.median([0.01, -0.05, 0.0, -0.2]))
    assert s["median_bias_over_W"] == pytest.approx(s["median_bias"] / 0.1)


def test_evaluate_run_refuses_existing_out_dir(tmp_path, monkeypatch):
    import evaluate_run

    monkeypatch.setattr(sys, "argv", ["evaluate_run.py", "--mode", "conditional", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="Refusing"):
        evaluate_run.main()
