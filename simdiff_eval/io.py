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


def load_real_reference_from_config(
    config_path: str | Path,
    max_slices: int | None = None,
) -> np.ndarray:
    """Load a reference normalized from the complete configured training set.

    Capping raw simulations before normalization changes inferred values such
    as ``center`` and ``xmax`` when those config entries are unset. Reference
    curves must instead use full-training normalization. Any plotting limit is
    therefore applied only after the complete set has been normalized.
    """
    import yaml

    with open(config_path) as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    try:
        return _load_real_reference_streaming(config, max_slices=max_slices)
    except (NotImplementedError, TypeError, ValueError) as exc:
        print(f"Streaming real-reference load unavailable; using full loader. Error: {exc}")
        full_reference = as_nchw(load_real_from_config(config_path, max_raw_samples=None))
        if max_slices is None or int(max_slices) <= 0 or len(full_reference) <= int(max_slices):
            return full_reference
        indices = np.linspace(0, len(full_reference) - 1, int(max_slices), dtype=np.int64)
        return np.asarray(full_reference[indices], dtype=np.float32).copy()


def _source_values(value: Any, n_sources: int, name: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        if len(value) != n_sources:
            raise ValueError(f"data.{name} has {len(value)} values for {n_sources} sources")
        return list(value)
    return [value] * n_sources


def _streaming_source_specs(data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    paths = data_cfg.get("img_path")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not isinstance(paths, (list, tuple)) or not paths:
        raise NotImplementedError("streaming references require one or more .npy paths")

    read_fns = _source_values(data_cfg.get("img_read_fn", "npy_read_fn"), len(paths), "img_read_fn")
    sample_counts = _source_values(data_cfg.get("n_samples"), len(paths), "n_samples")
    seeds = _source_values(data_cfg.get("seed"), len(paths), "seed")
    specs: list[dict[str, Any]] = []
    for path, read_fn, sample_count, seed in zip(paths, read_fns, sample_counts, seeds):
        if read_fn not in {None, "npy_read_fn"}:
            raise NotImplementedError(f"streaming reference does not support img_read_fn={read_fn!r}")
        array = np.load(path, mmap_mode="r")
        if array.ndim not in {3, 4}:
            raise ValueError(f"Expected raw .npy data with 3 or 4 dimensions, got {array.shape}")
        count = len(array) if sample_count is None else int(sample_count)
        if count < 0 or count > len(array):
            raise ValueError(f"Invalid n_samples={count} for {path} with {len(array)} entries")
        if seed is None:
            selected = np.arange(count, dtype=np.int64)
        else:
            selected = np.random.default_rng(seed).choice(len(array), size=count, replace=False)
        specs.append({"path": str(path), "array": array, "selected": np.asarray(selected, dtype=np.int64)})
    return specs


def _transformed_raw_chunks(
    specs: list[dict[str, Any]],
    *,
    use_log: bool,
    chunk_size: int = 4,
):
    for spec in specs:
        selected = spec["selected"]
        array = spec["array"]
        for start in range(0, len(selected), int(chunk_size)):
            indices = selected[start : start + int(chunk_size)]
            chunk = np.asarray(array[indices], dtype=np.float32)
            if use_log:
                chunk = np.log(chunk)
            yield chunk


def _full_normalization_stats(
    specs: list[dict[str, Any]],
    data_cfg: dict[str, Any],
) -> tuple[float | None, float | None]:
    transform = data_cfg.get("transform")
    use_log = bool(data_cfg.get("log", False)) or (
        isinstance(transform, (list, tuple)) and "log" in transform
    )
    normalization = data_cfg.get("normalization")
    norm_kwargs = dict(data_cfg.get("norm_kwargs") or {})
    if normalization not in {None, "none", "tanh", "centermax", "center-max", "centered_maxabs"}:
        raise NotImplementedError(f"streaming reference does not support normalization={normalization!r}")
    if normalization in {None, "none"}:
        return None, None

    center = norm_kwargs.get("center")
    if center is None:
        total = 0.0
        count = 0
        for chunk in _transformed_raw_chunks(specs, use_log=use_log):
            total += float(np.sum(chunk, dtype=np.float64))
            count += int(chunk.size)
        if count == 0:
            raise ValueError("Configured real reference contains no values")
        center = total / count

    xmax = norm_kwargs.get("xmax")
    if xmax is None:
        xmax = 0.0
        center32 = np.float32(center)
        for chunk in _transformed_raw_chunks(specs, use_log=use_log):
            xmax = max(xmax, float(np.max(np.abs(chunk - center32))))
    return float(center), float(xmax)


def _selected_reference_slices(
    specs: list[dict[str, Any]],
    data_cfg: dict[str, Any],
    max_slices: int | None,
) -> np.ndarray:
    reshape = data_cfg.get("reshape")
    two_dim = data_cfg.get("two_dim", True if reshape is None else reshape == "2d")
    if not (two_dim or reshape == "2d"):
        raise NotImplementedError("streaming reference currently supports 2D slice configs only")
    zthin = int(data_cfg.get("zthin", 1))
    if zthin < 1:
        raise ValueError("data.zthin must be >= 1")

    source_counts = []
    z_indices_by_source = []
    for spec in specs:
        array = spec["array"]
        z_indices = np.arange(0, array.shape[1], zthin, dtype=np.int64) if array.ndim == 4 else np.array([0])
        z_indices_by_source.append(z_indices)
        source_counts.append(len(spec["selected"]) * len(z_indices))
    total_slices = int(sum(source_counts))
    if total_slices == 0:
        raise ValueError("Configured real reference contains no 2D slices")

    limit = total_slices if max_slices is None or int(max_slices) <= 0 else min(total_slices, int(max_slices))
    global_indices = np.linspace(0, total_slices - 1, limit, dtype=np.int64)
    cumulative = np.cumsum(source_counts)
    output = []
    for global_index in global_indices:
        source_index = int(np.searchsorted(cumulative, global_index, side="right"))
        source_start = 0 if source_index == 0 else int(cumulative[source_index - 1])
        local_index = int(global_index) - source_start
        spec = specs[source_index]
        z_indices = z_indices_by_source[source_index]
        cube_position, z_position = divmod(local_index, len(z_indices))
        raw_index = int(spec["selected"][cube_position])
        if spec["array"].ndim == 4:
            image = spec["array"][raw_index, int(z_indices[z_position])]
        else:
            image = spec["array"][raw_index]
        output.append(np.asarray(image, dtype=np.float32))
    return np.stack(output, axis=0)


def _normalize_reference_slices(
    images: np.ndarray,
    data_cfg: dict[str, Any],
    center: float | None,
    xmax: float | None,
) -> np.ndarray:
    transform = data_cfg.get("transform")
    use_log = bool(data_cfg.get("log", False)) or (
        isinstance(transform, (list, tuple)) and "log" in transform
    )
    images = np.asarray(images, dtype=np.float32)
    if use_log:
        images = np.log(images)

    normalization = data_cfg.get("normalization")
    norm_kwargs = dict(data_cfg.get("norm_kwargs") or {})
    if normalization in {"tanh", "centermax", "center-max", "centered_maxabs"}:
        images = images - np.float32(center)
        images = images / np.float32(max(float(xmax), 1e-30))
    if normalization == "tanh":
        alpha = float(norm_kwargs.get("alpha", 1.0))
        beta = float(norm_kwargs.get("beta", 1.0))
        gamma = float(norm_kwargs.get("gamma", 1.0))
        delta = float(norm_kwargs.get("delta", 1.0))
        sigma = float(norm_kwargs.get("sigma", 1.0))
        mu = float(norm_kwargs.get("mu", 0.0))
        shifted = images - np.float32(mu)
        positive = alpha * np.tanh((gamma * shifted) / alpha)
        negative = beta * np.tanh((delta * shifted) / beta)
        images = np.where(shifted >= 0, positive, negative) * sigma
    return np.asarray(images[:, None], dtype=np.float32)


def _load_real_reference_streaming(
    config: dict[str, Any],
    max_slices: int | None,
) -> np.ndarray:
    data_cfg = config["data"]
    specs = _streaming_source_specs(data_cfg)
    center, xmax = _full_normalization_stats(specs, data_cfg)
    selected_slices = _selected_reference_slices(specs, data_cfg, max_slices)
    return _normalize_reference_slices(selected_slices, data_cfg, center, xmax)


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
