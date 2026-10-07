"""Independent NumPy audit of cosmodiff's native slice-first data loader."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CONTRACT = "native_select_then_zthin_then_log_centermax_tanh_v1"


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _source_values(value, count, name):
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"data.{name} must have one value per source")
        return list(value)
    return [value] * count


def load_native_training_reference(config):
    """Read exactly the retained slices, in native source/volume/z order."""
    data = config["data"]
    if config.get("global", {}).get("dtype", "float32") != "float32":
        raise ValueError("the native-reference contract requires float32")
    if data.get("reshape", "2d") != "2d" or data.get("normalization") != "tanh":
        raise ValueError("the screen reference requires native 2D/tanh data")
    if data.get("transform") != ["log"] or data.get("log", False):
        raise ValueError("the screen reference requires the native ['log'] transform")
    if config.get("augmentations"):
        raise ValueError("the screen requires no augmentations")
    paths = data["img_path"]
    paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
    readers = _source_values(data.get("img_read_fn"), len(paths), "img_read_fn")
    counts = _source_values(data.get("n_samples"), len(paths), "n_samples")
    seeds = _source_values(data.get("seed"), len(paths), "seed")
    thins = _source_values(data.get("zthin", 1), len(paths), "zthin")
    raw_slices, selections = [], []
    for path, reader, count, seed, thin in zip(paths, readers, counts, seeds, thins):
        if reader != "npy_read_fn" or isinstance(thin, bool) or int(thin) != thin or thin < 1:
            raise ValueError("reference requires npy_read_fn and positive integer zthin")
        raw = np.load(Path(path), mmap_mode="r", allow_pickle=False)
        if raw.ndim != 4:
            raise ValueError(f"native 2D data requires (volume,z,h,w), found {raw.shape}")
        n = len(raw) if count is None else int(count)
        if isinstance(count, bool) or (count is not None and n != count) or not 0 <= n <= len(raw):
            raise ValueError(f"invalid native n_samples={count}")
        selected = (np.arange(n) if seed is None or count is None
                    else np.random.default_rng(seed).choice(len(raw), size=n, replace=False))
        kept = np.asarray(raw[selected][:, ::int(thin)], dtype=np.float32)
        raw_slices.append(kept.reshape(-1, 1, *raw.shape[-2:]).copy())
        selections.append({"path": str(path), "volume_indices": selected.tolist(),
                           "z_indices": list(range(0, raw.shape[1], int(thin)))})
    slices = np.concatenate(raw_slices, axis=0)
    if not len(slices) or not np.isfinite(slices).all() or np.any(slices <= 0):
        raise ValueError("native log data must contain nonempty finite positive slices")
    transformed = np.log(slices)
    kwargs = dict(data.get("norm_kwargs") or {})
    center = kwargs.get("center")
    if center is None:
        center = float(transformed.mean(dtype=np.float64))
    normalized = transformed - np.float32(center)
    xmax = kwargs.get("xmax")
    if xmax is None:
        xmax = float(np.abs(normalized).max())
    if not np.isfinite(center) or not np.isfinite(xmax) or xmax <= 0:
        raise ValueError("native center/max normalization is invalid")
    shifted = normalized / np.float32(xmax) - np.float32(kwargs.get("mu", 0.0))
    alpha, beta = float(kwargs.get("alpha", 1.0)), float(kwargs.get("beta", 1.0))
    gamma, delta = float(kwargs.get("gamma", 1.0)), float(kwargs.get("delta", 1.0))
    sigma = float(kwargs.get("sigma", 1.0))
    if not all(np.isfinite(x) and x > 0 for x in (alpha, beta, gamma, delta, sigma)):
        raise ValueError("native tanh parameters must be finite and positive")
    positive = alpha * np.tanh((gamma * shifted) / alpha)
    negative = beta * np.tanh((delta * shifted) / beta)
    reference = np.asarray(np.where(shifted >= 0, positive, negative) * sigma, dtype=np.float32)
    return reference, {
        "contract": CONTRACT, "shape": list(reference.shape), "dtype": str(reference.dtype),
        "selected_raw_sha256": array_hash(slices), "reference_sha256": array_hash(reference),
        "center": float(center), "xmax": float(xmax), "selections": selections,
    }


def audit_native_dataset(parsed, reference, metadata, torch):
    dataset = parsed["data"]
    actual = dataset.arrays.detach().cpu().numpy()
    if actual.shape != reference.shape or actual.dtype != np.float32:
        raise ValueError("native training tensor shape/dtype mismatch")
    if not np.isfinite(actual).all():
        raise ValueError("native training tensors contain nonfinite values")
    max_delta = float(np.max(np.abs(actual - reference)))
    if not np.allclose(actual, reference, rtol=0, atol=2e-6):
        raise ValueError(f"native tensors disagree with slice-first reference: delta={max_delta}")
    labels = dataset.labels
    if (labels is None or labels.dtype != torch.long or tuple(labels.shape) != (len(reference),)
            or bool(torch.any(labels != 0).item())):
        raise ValueError("null class labels must be all-zero long[N]")
    normalization = parsed.get("norm")
    fitted = dict(getattr(normalization, "kwargs", {}))
    for key in ("center", "xmax"):
        value = fitted.get(key)
        if value is None or not np.isclose(value, metadata[key], rtol=0, atol=2e-6):
            raise ValueError(f"native fitted {key} differs from reference")
    return {
        "status": "matched", "contract": CONTRACT,
        "training_tensor_sha256": array_hash(actual),
        "training_reference_sha256": array_hash(reference),
        "selected_raw_sha256": metadata["selected_raw_sha256"],
        "reference_max_abs_delta": max_delta,
        "native_fitted_normalization": {key: float(fitted[key]) for key in ("center", "xmax")},
        "shape": list(actual.shape), "dtype": str(actual.dtype), "null_labels": "all_zero_long",
    }
