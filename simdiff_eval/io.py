"""Input helpers for lightweight CAMELS diffusion evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_npy(path: str | Path) -> np.ndarray:
    """Load an ``.npy`` array without modifying values."""
    return np.asarray(np.load(Path(path), mmap_mode="r"))


def as_nchw(images: np.ndarray) -> np.ndarray:
    """Return images as ``(N, C, H, W)``.

    Supported inputs:
    - ``(N, H, W)``
    - ``(N, 1, H, W)``
    - ``(N, Z, H, W)`` after slicing/reshaping elsewhere
    """
    arr = np.asarray(images)
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr
    raise ValueError(f"Expected (N,H,W) or (N,1,H,W), got shape {arr.shape}.")


def cubes_to_slices(cubes: np.ndarray, zthin: int = 1) -> np.ndarray:
    """Convert ``(N, Z, H, W)`` cubes into ``(N * Z_kept, 1, H, W)`` slices."""
    arr = np.asarray(cubes)
    if arr.ndim != 4:
        raise ValueError(f"Expected raw cubes shaped (N,Z,H,W), got {arr.shape}.")
    if zthin < 1:
        raise ValueError("zthin must be >= 1.")
    arr = arr[:, ::zthin]
    return arr.reshape(-1, 1, *arr.shape[-2:])


def load_real_from_config(config_path: str | Path, max_raw_samples: int | None = None) -> np.ndarray:
    """Load real data using ``cosmodiff`` config normalization.

    This requires the local ``cosmo_diffusion`` checkout to be on
    ``PYTHONPATH`` or importable from the project root.
    """
    import yaml

    from cosmodiff import utils

    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    config = dict(config)
    config.setdefault("global", {})["device"] = "cpu"
    config.setdefault("data", {})["keep_on_cpu"] = True
    if max_raw_samples is not None:
        data_cfg = config.setdefault("data", {})
        current = data_cfg.get("n_samples")
        data_cfg["n_samples"] = int(max_raw_samples) if current is None else min(int(current), int(max_raw_samples))
    try:
        parsed = utils.parse_config_data(config)
    except ValueError as exc:
        if "Unknown normalization mode 'tanh'" not in str(exc):
            raise
        return _load_real_tanh_from_config(config, utils)

    if isinstance(parsed, dict):
        dataset = parsed["data"]
    elif isinstance(parsed, tuple):
        dataset = parsed[0]
    else:
        dataset = parsed
    return dataset.arrays.detach().cpu().numpy()


def _load_real_tanh_from_config(config: dict[str, Any], utils_module: Any) -> np.ndarray:
    """Load real data for older cosmodiff checkouts missing tanh normalization."""
    import torch

    data_cfg = config["data"]
    global_cfg = config.get("global", {})
    dtype = getattr(torch, global_cfg.get("dtype", "float32"))
    img_read_fn = getattr(utils_module, data_cfg["img_read_fn"])
    images = img_read_fn(data_cfg["img_path"])

    n_samples = data_cfg.get("n_samples", None)
    seed = data_cfg.get("seed", None)
    if n_samples is not None:
        if seed is None:
            idx = slice(n_samples)
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(images), size=n_samples, replace=False)
        images = images[idx]
    else:
        images = images[:]

    images = torch.as_tensor(np.array(images, copy=True), device="cpu", dtype=dtype)
    if data_cfg.get("log", False):
        images = images.log()

    norm_kwargs = dict(data_cfg.get("norm_kwargs") or {})
    center = norm_kwargs.get("center", None)
    if center is None:
        center = images.mean()
    elif not torch.is_tensor(center):
        center = torch.as_tensor(center, device=images.device, dtype=images.dtype)
    images = images - center

    xmax = norm_kwargs.get("xmax", None)
    if xmax is None:
        xmax = images.abs().max()
    elif not torch.is_tensor(xmax):
        xmax = torch.as_tensor(xmax, device=images.device, dtype=images.dtype)
    images = images / xmax

    alpha = float(norm_kwargs.get("alpha", 1.0))
    beta = float(norm_kwargs.get("beta", 1.0))
    gamma = float(norm_kwargs.get("gamma", 1.0))
    delta = float(norm_kwargs.get("delta", 1.0))
    sigma = float(norm_kwargs.get("sigma", 1.0))
    mu = float(norm_kwargs.get("mu", 0.0))

    shifted = images - mu
    pos = alpha * torch.tanh((gamma * shifted) / alpha)
    neg = beta * torch.tanh((delta * shifted) / beta)
    images = torch.where(shifted >= 0, pos, neg) * sigma

    if data_cfg.get("two_dim", True):
        zthin = int(data_cfg.get("zthin", 1))
        images = images[:, ::zthin]
        images = images.reshape(-1, 1, *images.shape[-2:])
    else:
        images = images.unsqueeze(1)

    return images.detach().cpu().numpy()
