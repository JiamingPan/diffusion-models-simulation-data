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


def load_real_from_config(config_path: str | Path) -> np.ndarray:
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
    dataset = utils.parse_config_data(config)
    return dataset.arrays.detach().cpu().numpy()
