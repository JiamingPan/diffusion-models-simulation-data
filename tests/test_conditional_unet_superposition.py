import numpy as np
import pytest

from simdiff_eval.conditional_unet_superposition import (
    centered_cosine, interpolation_labels, lowpass_maps, mutually_nearest_pairs,
    nnls_topk_curve, random_basis_indices, superposition_controls_run,
    theta_interpolation_analysis, training_theta_analysis,
)

RNG = np.random.default_rng(0)


def smooth_field(seed, size=16):
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(size, size))
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    f = np.fft.fft2(white) / np.maximum(kk, 1.0) ** 1.5
    return np.fft.ifft2(f).real.astype(np.float64)


def test_centered_cosine_identity_and_orthogonal():
    a = np.array([[1., -1.], [1., -1.]])
    b = np.array([[1., 1.], [-1., -1.]])
    assert centered_cosine(a, a + 5) == pytest.approx(1)
    assert centered_cosine(a, b) == pytest.approx(0)


def test_lowpass_periodic_and_identity_at_zero():
    x = np.stack([smooth_field(1), smooth_field(2)])
    np.testing.assert_allclose(lowpass_maps(x, 0), x)
    y = lowpass_maps(x, 2.0)
    assert y.shape == x.shape
    assert y.var() < x.var()
    np.testing.assert_allclose(y.mean(axis=(1, 2)), x.mean(axis=(1, 2)), atol=1e-10)


def test_random_basis_excludes_nearest_and_is_reproducible():
    idx = random_basis_indices(40, np.arange(16), 16, seed=3)
    assert len(idx) == 16 and not np.intersect1d(idx, np.arange(16)).size
    np.testing.assert_array_equal(idx, random_basis_indices(40, np.arange(16), 16, seed=3))
    assert len(random_basis_indices(20, np.arange(16), 16, seed=3)) == 4
    with pytest.raises(ValueError):
        random_basis_indices(16, np.arange(16), 16, seed=3)


def test_topk_curve_exact_two_map_blend_is_flat_after_two():
    a, b, c = smooth_field(11), smooth_field(12), smooth_field(13)
    target = 0.7 * a + 0.3 * b
    rows = nnls_topk_curve(target, np.stack([a, b, c]), ks=(1, 2))
    ks = [r["k"] for r in rows]
    assert ks == [1, 2, 3]
    assert rows[0]["single_map_cos2"] == pytest.approx(centered_cosine(target, a) ** 2)
    assert rows[0]["r2"] < rows[1]["r2"]
    assert rows[1]["r2"] == pytest.approx(1, abs=1e-8)
    assert rows[2]["r2"] == pytest.approx(1, abs=1e-8)


def test_mutually_nearest_pairs_and_labels():
    theta = np.array([[0., 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0],
                      [5., 0, 0, 0, 0, 0], [5.2, 0, 0, 0, 0, 0],
                      [20., 0, 0, 0, 0, 0]])
    ids = np.array([10, 11, 12, 13, 14])
    pairs = mutually_nearest_pairs(theta, ids, np.ones(6), n_pairs=2)
    assert [(p["sim_a"], p["sim_b"]) for p in pairs] == [(10, 11), (12, 13)]
    assert pairs[0]["distance"] == pytest.approx(0.1)
    labels, index = interpolation_labels(theta, pairs, np.array([0, .5, 1]), n_seeds=2)
    assert labels.shape == (2 * 3 * 2, 6)
    np.testing.assert_allclose(labels[0], theta[0])
    np.testing.assert_allclose(labels[2], [0.05, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(labels[5], theta[1])
    assert index[5] == {"pair": 0, "s_index": 2, "s": 1.0, "seed": 1, "row": 5}
    with pytest.raises(ValueError):
        mutually_nearest_pairs(theta, ids, np.ones(6), n_pairs=3)


def test_interpolation_analysis_recovers_linear_morph():
    a, b = smooth_field(21), smooth_field(22)
    s_values = np.array([0., .5, 1.])
    pairs = [{"row_a": 0, "row_b": 1}]
    _, index = interpolation_labels(np.zeros((2, 6)), pairs, s_values, n_seeds=2)
    samples = np.stack([(1 - r["s"]) * a + r["s"] * b + 0.3 for r in index])
    rows = theta_interpolation_analysis(samples, a[None], b[None], index, s_values)
    assert [r["s"] for r in rows] == [0., .5, 1.]
    assert rows[0]["cos_a"] == pytest.approx(1) and rows[2]["cos_b"] == pytest.approx(1)
    assert rows[1]["weight_a"] == pytest.approx(.5, abs=1e-6)
    assert all(r["two_map_r2"] == pytest.approx(1, abs=1e-8) for r in rows)
    assert all(r["draw_to_draw_cosine"] == pytest.approx(1) for r in rows)


def test_training_theta_analysis_detects_copies():
    own = np.stack([smooth_field(31), smooth_field(32)])
    samples = np.stack([own[0], own[0] + 1, own[1], smooth_field(33)])
    rows = training_theta_analysis(samples, own, n_seeds=2)
    assert rows[0]["cos_to_own_map"] == pytest.approx(1)
    assert rows[0]["draw_to_draw_cosine"] == pytest.approx(1)
    assert rows[1]["cos_to_own_map"] < 0.9


def test_superposition_controls_run_shapes_and_controls():
    size, n_train, n_held, k = 8, 12, 2, 3
    train = np.stack([np.exp(smooth_field(100 + i, size)) for i in range(n_train)])
    theta = RNG.normal(size=(n_train, 6))
    ids = np.arange(n_train)
    heldout = np.array([900, 901])
    requested = theta[:2] + 0.01
    norm = {"center": 0.0, "xmax": 1.0}
    from simdiff_eval.conditional_unet_diagnostics import normalize_raw_hi
    model_space = normalize_raw_hi(train[:2], norm)
    gen = np.stack([model_space[0]] * k + [model_space[1]] * k)
    real = np.stack([np.stack([smooth_field(200 + j, size) for j in range(k)]) for _ in range(n_held)])
    rows = superposition_controls_run(gen, train, theta, ids, heldout, requested, np.ones(6), real,
                                      norm, k=k, k_cosmologies=4, cap=4, sigmas=(1.0,), seed=1)
    tests = {r["test"] for r in rows}
    assert tests == {"parameter-nearest top-k", "random basis (excludes nearest)",
                     "low-pass parameter-nearest"}
    gen_top1 = [r for r in rows if r["kind"] == "generated" and r["test"] == "parameter-nearest top-k" and r["k"] == 1]
    assert len(gen_top1) == n_held
    # The generated map IS the nearest training map (in model space), so k=1 explains it fully.
    assert all(r["r2"] == pytest.approx(1, abs=1e-6) for r in gen_top1)
    rand = [r for r in rows if r["kind"] == "generated" and r["test"].startswith("random")]
    assert all(r["r2"] < 0.9 for r in rand)
