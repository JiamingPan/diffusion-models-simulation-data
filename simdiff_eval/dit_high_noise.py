"""Numerical helpers for the DiT high-noise failure diagnostic."""

from __future__ import annotations

from typing import Any

import numpy as np


def snr_and_v_weight(alphas_cumprod: Any, gamma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Return scheduler SNR and the Min-SNR weight used for v-prediction."""
    alpha_bar = np.asarray(
        alphas_cumprod.detach().cpu() if hasattr(alphas_cumprod, "detach") else alphas_cumprod,
        dtype=np.float64,
    )
    if alpha_bar.ndim != 1 or alpha_bar.size < 2:
        raise ValueError("alphas_cumprod must be a one-dimensional schedule")
    if not np.isfinite(alpha_bar).all() or np.any(alpha_bar < 0.0) or np.any(alpha_bar > 1.0):
        raise ValueError("alphas_cumprod must contain finite probabilities")
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    denominator = np.maximum(1.0 - alpha_bar, np.finfo(np.float64).tiny)
    snr = alpha_bar / denominator
    weight = np.minimum(snr, gamma) / (snr + 1.0)
    return snr, weight


def x0_from_v(x_t: Any, v_prediction: Any, alpha_bar: Any) -> Any:
    """Recover x0 from v using x_t=alpha*x0+sigma*eps."""
    alpha = alpha_bar.sqrt() if hasattr(alpha_bar, "sqrt") else np.sqrt(alpha_bar)
    sigma_source = 1.0 - alpha_bar
    sigma = sigma_source.sqrt() if hasattr(sigma_source, "sqrt") else np.sqrt(sigma_source)
    return alpha * x_t - sigma * v_prediction


def batch_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity for equally shaped image batches."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim < 2:
        raise ValueError(f"batch shapes must agree and include a batch axis: {x.shape}, {y.shape}")
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return np.divide(
        np.sum(x * y, axis=1),
        denominator,
        out=np.zeros(len(x), dtype=np.float64),
        where=denominator > 0.0,
    )


def batch_nmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-image MSE divided by the target's per-image mean square."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim < 2:
        raise ValueError(f"batch shapes must agree and include a batch axis: {pred.shape}, {truth.shape}")
    axes = tuple(range(1, pred.ndim))
    mse = np.mean((pred - truth) ** 2, axis=axes)
    scale = np.mean(truth**2, axis=axes)
    return np.divide(mse, scale, out=np.full_like(mse, np.inf), where=scale > 0.0)


def choose_inference_begin(timesteps: Any, requested_timestep: int) -> tuple[int, int]:
    """Choose the closest inference-schedule timestep and return index/value."""
    values = np.asarray(
        timesteps.detach().cpu() if hasattr(timesteps, "detach") else timesteps,
        dtype=np.float64,
    ).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("inference timesteps must be finite and non-empty")
    index = int(np.argmin(np.abs(values - int(requested_timestep))))
    return index, int(round(float(values[index])))
