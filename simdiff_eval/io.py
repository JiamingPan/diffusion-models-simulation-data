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
    """Load real data using the run config normalization.

    Prefer ``cosmodiff`` when importable.  If the HPC Python environment has a
    broken ``diffusers``/``transformers`` mix, fall back to a small local
    loader that supports the normalization-fix configs used in these notebooks.
    """
    import yaml

    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    config = dict(config)
    config.setdefault("global", {})["device"] = "cpu"
    config.setdefault("data", {})["keep_on_cpu"] = True
    if max_raw_samples is not None:
        data_cfg = config.setdefault("data", {})
        current = data_cfg.get("n_samples")
        img_path = data_cfg.get("img_path")
        data_cfg["n_samples"] = _cap_n_samples(current, max_raw_samples, img_path)

    if isinstance(config.get("data", {}).get("img_path"), (list, tuple)):
        return _load_real_tanh_from_config(config, utils_module=None)

    try:
        from cosmodiff import utils
    except Exception as exc:
        print(f"cosmodiff import failed while loading real data; using local config loader. Error: {exc}")
        return _load_real_tanh_from_config(config, utils_module=None)

    try:
        parsed = utils.parse_config_data(config)
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        if (
            isinstance(exc, ValueError)
            and "Unknown normalization mode 'tanh'" not in str(exc)
        ):
            raise
        print(f"cosmodiff parse_config_data failed; using local config loader. Error: {exc}")
        return _load_real_tanh_from_config(config, utils)

    if isinstance(parsed, dict):
        dataset = parsed["data"]
    elif isinstance(parsed, tuple):
        dataset = parsed[0]
    else:
        dataset = parsed
    return dataset.arrays.detach().cpu().numpy()


def _cap_n_samples(current: Any, max_raw_samples: int, img_path: Any = None) -> int | list[int]:
    """Cap config ``n_samples`` while preserving list-of-file semantics."""
    max_raw_samples = int(max_raw_samples)
    if max_raw_samples < 1:
        raise ValueError("max_raw_samples must be >= 1.")

    if isinstance(current, (list, tuple)):
        caps = [int(x) for x in current]
        counts = [0 for _ in caps]
        remaining = min(max_raw_samples, sum(caps))
        while remaining > 0:
            progressed = False
            for i, cap in enumerate(caps):
                if remaining == 0:
                    break
                if counts[i] >= cap:
                    continue
                counts[i] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break
        return counts

    if current is None and isinstance(img_path, (list, tuple)):
        n = len(img_path)
        counts = [0 for _ in range(n)]
        remaining = max_raw_samples
        while remaining > 0:
            for i in range(n):
                if remaining == 0:
                    break
                counts[i] += 1
                remaining -= 1
        return counts

    return max_raw_samples if current is None else min(int(current), max_raw_samples)


def _read_images(data_cfg: dict[str, Any], utils_module: Any | None) -> np.ndarray:
    def read_one(path: Any, read_fn_name: str | None) -> np.ndarray:
        if isinstance(path, np.ndarray):
            return path
        if utils_module is not None and read_fn_name and hasattr(utils_module, read_fn_name):
            return getattr(utils_module, read_fn_name)(path)
        if read_fn_name in {None, "npy_read_fn"}:
            return np.load(path, mmap_mode="r")
        if read_fn_name == "txt_read_fn":
            return np.loadtxt(path)
        raise ValueError(f"Local loader does not know img_read_fn={read_fn_name!r}.")

    img_path = data_cfg["img_path"]
    img_read_fn = data_cfg.get("img_read_fn", "npy_read_fn")
    n_samples = data_cfg.get("n_samples", None)
    seed = data_cfg.get("seed", None)

    if isinstance(img_path, (list, tuple)):
        n = len(img_path)
        read_fns = img_read_fn if isinstance(img_read_fn, (list, tuple)) else [img_read_fn] * n
        samples = n_samples if isinstance(n_samples, (list, tuple)) else [n_samples] * n
        seeds = seed if isinstance(seed, (list, tuple)) else [seed] * n
        arrays = []
        for path, read_fn, nsamp, one_seed in zip(img_path, read_fns, samples, seeds):
            arr = read_one(path, read_fn)
            arrays.append(_select_raw_samples(arr, nsamp, one_seed))
        return np.concatenate(arrays, axis=0)

    images = read_one(img_path, img_read_fn)
    return _select_raw_samples(images, n_samples, seed)


def _select_raw_samples(images: np.ndarray, n_samples: int | None, seed: int | None) -> np.ndarray:
    if n_samples is not None:
        if seed is None:
            idx: slice | np.ndarray = slice(int(n_samples))
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(images), size=int(n_samples), replace=False)
        images = images[idx]
    else:
        images = images[:]
    return np.asarray(images, dtype=np.float32).copy()


def _load_real_tanh_from_config(config: dict[str, Any], utils_module: Any | None) -> np.ndarray:
    """Local real-data loader for old/new normalization-fix configs."""
    data_cfg = config["data"]
    images = _read_images(data_cfg, utils_module)

    transform = data_cfg.get("transform", None)
    use_log = bool(data_cfg.get("log", False)) or (
        isinstance(transform, (list, tuple)) and "log" in transform
    )
    if use_log:
        images = np.log(images)

    normalization = data_cfg.get("normalization", None)
    norm_kwargs = dict(data_cfg.get("norm_kwargs") or {})
    center = norm_kwargs.get("center", None)
    xmax = norm_kwargs.get("xmax", None)
    if normalization in {"tanh", "centermax", "center-max", "centered_maxabs"}:
        if center is None:
            center = float(images.mean())
        images = images - np.float32(center)
        if xmax is None:
            xmax = float(np.abs(images).max())
        images = images / np.float32(max(float(xmax), 1e-30))

    if normalization == "tanh":
        alpha = float(norm_kwargs.get("alpha", 1.0))
        beta = float(norm_kwargs.get("beta", 1.0))
        gamma = float(norm_kwargs.get("gamma", 1.0))
        delta = float(norm_kwargs.get("delta", 1.0))
        sigma = float(norm_kwargs.get("sigma", 1.0))
        mu = float(norm_kwargs.get("mu", 0.0))

        shifted = images - np.float32(mu)
        pos = alpha * np.tanh((gamma * shifted) / alpha)
        neg = beta * np.tanh((delta * shifted) / beta)
        images = np.where(shifted >= 0, pos, neg) * sigma
    elif normalization in {None, "none", "centermax", "center-max", "centered_maxabs"}:
        pass
    elif normalization in {"minmax", "min-max"}:
        xmin = norm_kwargs.get("xmin", None)
        xmax = norm_kwargs.get("xmax", None)
        if xmin is None:
            xmin = float(images.min())
        if xmax is None:
            xmax = float(images.max())
        images = (images - np.float32(xmin)) * (2.0 / np.float32(max(float(xmax), 1e-30))) - 1.0
    else:
        raise ValueError(f"Local loader does not support normalization={normalization!r}.")

    reshape = data_cfg.get("reshape", None)
    two_dim = data_cfg.get("two_dim", True if reshape is None else reshape == "2d")
    if two_dim or reshape == "2d":
        zthin = int(data_cfg.get("zthin", 1))
        images = images[:, ::zthin]
        images = images.reshape(-1, 1, *images.shape[-2:])
    elif reshape == "3d" or two_dim is False:
        images = images[:, None]
    else:
        images = as_nchw(images)

    return np.asarray(images, dtype=np.float32)
