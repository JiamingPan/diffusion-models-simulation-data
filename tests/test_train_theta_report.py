import numpy as np
import pytest

from simdiff_eval.train_theta_report import train_theta_report


def field(seed, size=12):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(size, size))
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    return np.fft.ifft2(np.fft.fft2(w) / np.maximum(kk, 1) ** 1.5).real


def test_exact_copies_score_one_and_nearest_is_own():
    own = np.stack([field(i) for i in range(5)])
    samples = np.repeat(own, 3, axis=0) + 0.2      # copies with a constant offset
    truth = np.linspace(.1, .5, 5)
    r = train_theta_report(samples, own, 3, truth,
                           predict_omega=lambda m: np.full(len(m), .3))
    s = r["summary"]
    assert s["cos_to_own_map"]["median"] == pytest.approx(1)
    assert s["max_cos_any_training_map"]["median"] == pytest.approx(1)
    assert s["nearest_is_own_fraction"]["median"] == 1
    assert s["within_cosmology_draw_cosine"]["median"] == pytest.approx(1)
    assert s["paired_generated_minus_real"]["median"] == pytest.approx(0)
    assert s["generated_bias"]["n"] == 5
    assert r["per_cosmology"][0]["generated_bias"] == pytest.approx(.3 - .1)


def test_novel_fields_have_low_cosine_and_wrong_nearest():
    own = np.stack([field(i) for i in range(4)])
    samples = np.stack([field(100 + i) for i in range(8)])
    r = train_theta_report(samples, own, 2, np.zeros(4))
    assert r["summary"]["cos_to_own_map"]["median"] < .5
    assert r["summary"]["within_cosmology_draw_cosine"]["median"] < .5
    assert "generated_bias" not in r["summary"]


def test_rejects_misaligned_inputs():
    own = np.stack([field(i) for i in range(4)])
    with pytest.raises(ValueError):
        train_theta_report(np.repeat(own, 2, axis=0), own, 3, np.zeros(4))


def test_multiplicity_groups_use_best_own_map_and_median_real_probe():
    own = np.stack([field(i) for i in range(6)])        # 3 cosmologies x 2 maps
    groups = np.array([0, 0, 1, 1, 2, 2])
    # each cosmology's draws copy its SECOND map
    samples = np.stack([own[1], own[1], own[3], own[3], own[5], own[5]])
    truth = np.array([.2, .3, .4])
    probe = lambda m: np.array([0.1 * (k + 1) for k in range(len(m))])  # distinct per input map
    r = train_theta_report(samples, own, 2, truth, predict_omega=probe, group_index=groups)
    s = r["summary"]
    assert s["cos_to_own_map"]["median"] == pytest.approx(1)
    assert s["nearest_is_own_fraction"]["median"] == 1
    assert r["per_cosmology"][0]["n_training_maps"] == 2
    # real control = median over the cosmology's 2 maps: (0.1+0.2)/2 for cosmology 0
    assert r["per_cosmology"][0]["real_omega_median"] == pytest.approx(0.15)
