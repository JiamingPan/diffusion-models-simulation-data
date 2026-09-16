import numpy as np
import pytest

from simdiff_eval.dit_high_noise import (
    batch_cosine,
    batch_nmse,
    choose_inference_begin,
    snr_and_v_weight,
    x0_from_v,
)


def test_v_prediction_x0_inverse_is_exact():
    rng = np.random.default_rng(4)
    x0 = rng.normal(size=(3, 1, 4, 4))
    noise = rng.normal(size=x0.shape)
    alpha_bar = np.asarray(0.36)
    alpha, sigma = np.sqrt(alpha_bar), np.sqrt(1.0 - alpha_bar)
    x_t = alpha * x0 + sigma * noise
    velocity = alpha * noise - sigma * x0
    np.testing.assert_allclose(x0_from_v(x_t, velocity, alpha_bar), x0, atol=1e-12)


def test_v_weight_vanishes_at_zero_terminal_snr():
    snr, weight = snr_and_v_weight(np.array([0.9, 0.5, 0.01, 0.0]), gamma=5.0)
    assert snr[-1] == 0.0
    assert weight[-1] == 0.0
    assert weight[-2] < 0.02


def test_batch_metrics_distinguish_exact_and_zero_predictions():
    target = np.arange(1, 33, dtype=float).reshape(2, 1, 4, 4)
    np.testing.assert_allclose(batch_cosine(target, target), 1.0)
    np.testing.assert_allclose(batch_nmse(target, target), 0.0)
    np.testing.assert_allclose(batch_cosine(np.zeros_like(target), target), 0.0)
    np.testing.assert_allclose(batch_nmse(np.zeros_like(target), target), 1.0)


def test_choose_inference_begin_uses_closest_schedule_value():
    index, value = choose_inference_begin([499, 449, 399, 299, 0], 410)
    assert (index, value) == (2, 399)


def test_invalid_weight_schedule_fails_closed():
    with pytest.raises(ValueError, match="probabilities"):
        snr_and_v_weight(np.array([1.1, 0.5]))
