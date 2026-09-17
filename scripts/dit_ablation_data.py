"""Independent NumPy reference matching native cosmodiff's slice-first loader.

The frozen legacy evaluation reader fits center/max over unthinned volumes.
Native load_data first selects volumes, thins z, concatenates the retained
slices, and only then fits log/center/max/tanh. Never replace the native tensor
with this reference or change the native training recipe to match legacy IO.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CONTRACT = "native_select_then_zthin_then_log_centermax_tanh_v1"


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_values(value, count, name):
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"data.{name} must have one value per source")
        return list(value)
    return [value] * count


def load_native_training_reference(config):
    """Read exactly the native retained slices, in native source/volume/z order."""
    data = config["data"]
    if config.get("global", {}).get("dtype", "float32") != "float32":
        raise ValueError("the native-reference contract requires float32")
    if data.get("reshape", "2d") != "2d" or data.get("normalization") != "tanh":
        raise ValueError("the ablation reference requires native 2D/tanh data")
    if data.get("transform") != ["log"] or data.get("log", False):
        raise ValueError("the ablation reference requires the native ['log'] transform")
    if config.get("augmentations"):
        raise ValueError("the ablation reference requires no augmentations")
    paths = data["img_path"]
    paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
    if not paths:
        raise ValueError("no configured data sources")
    readers = source_values(data.get("img_read_fn"), len(paths), "img_read_fn")
    counts = source_values(data.get("n_samples"), len(paths), "n_samples")
    seeds = source_values(data.get("seed"), len(paths), "seed")
    thins = source_values(data.get("zthin", 1), len(paths), "zthin")
    raw_slices, selections = [], []
    for path, reader, count, seed, thin in zip(paths, readers, counts, seeds, thins):
        if reader != "npy_read_fn" or isinstance(thin, bool) or int(thin) != thin or thin < 1:
            raise ValueError("native reference requires npy_read_fn and positive integer zthin")
        raw = np.load(Path(path), mmap_mode="r", allow_pickle=False)
        if raw.ndim != 4:
            raise ValueError(f"native 2D data requires (volume,z,h,w), found {raw.shape}")
        n = len(raw) if count is None else int(count)
        if isinstance(count, bool) or (count is not None and n != count) or not 0 <= n <= len(raw):
            raise ValueError(f"invalid native n_samples={count}")
        # Native ignores seed when n_samples is None.
        selected = (np.arange(n) if seed is None or count is None
                    else np.random.default_rng(seed).choice(len(raw), size=n, replace=False))
        kept = np.asarray(raw[selected][:, ::int(thin)], dtype=np.float32)
        raw_slices.append(kept.reshape(-1, 1, *raw.shape[-2:]).copy())
        selections.append({"path": str(path), "volume_indices": selected.tolist(),
                           "z_indices": list(range(0, raw.shape[1], int(thin)))})
    slices = np.concatenate(raw_slices, axis=0)
    if not len(slices) or not np.isfinite(slices).all() or np.any(slices <= 0):
        raise ValueError("native log data must contain nonempty finite positive retained slices")
    transformed = np.log(slices)
    kwargs = dict(data.get("norm_kwargs") or {})
    center = kwargs.get("center")
    if center is None:
        # float64 reduction avoids a reference-only reduction-order dependency;
        # the elementwise audit permits only <=2e-6 Torch/NumPy rounding.
        center = float(transformed.mean(dtype=np.float64))
    normalized = transformed - np.float32(center)
    xmax = kwargs.get("xmax")
    if xmax is None:
        xmax = float(np.abs(normalized).max())
    if not np.isfinite(center) or not np.isfinite(xmax) or xmax <= 0:
        raise ValueError("native center/max normalization must be finite and nondegenerate")
    normalized = normalized / np.float32(xmax)
    shifted = normalized - np.float32(kwargs.get("mu", 0.0))
    alpha, beta = float(kwargs.get("alpha", 1.0)), float(kwargs.get("beta", 1.0))
    gamma, delta = float(kwargs.get("gamma", 1.0)), float(kwargs.get("delta", 1.0))
    sigma = float(kwargs.get("sigma", 1.0))
    if not all(np.isfinite(x) and x > 0 for x in (alpha, beta, gamma, delta, sigma)):
        raise ValueError("native tanh parameters must be finite and positive")
    positive = alpha * np.tanh((gamma * shifted) / alpha)
    negative = beta * np.tanh((delta * shifted) / beta)
    reference = np.asarray(np.where(shifted >= 0, positive, negative) * sigma, dtype=np.float32)
    if not np.isfinite(reference).all():
        raise ValueError("native reference contains nonfinite values")
    return reference, {"contract": CONTRACT, "shape": list(reference.shape),
        "dtype": str(reference.dtype), "selected_raw_sha256": array_hash(slices),
        "reference_sha256": array_hash(reference), "center": float(center),
        "xmax": float(xmax), "selections": selections}


def audit_native_dataset(parsed, reference, metadata, torch):
    """Check every element and label, leaving the native dataset untouched."""
    dataset = parsed["data"]
    actual = dataset.arrays.detach().cpu().numpy()
    diagnostic = {"shape": list(actual.shape), "dtype": str(actual.dtype),
                  "expected_shape": list(reference.shape), "expected_dtype": "float32"}
    if actual.shape != reference.shape or actual.dtype != np.float32:
        raise ValueError(f"native training tensor shape/dtype mismatch: {diagnostic}")
    if not np.isfinite(actual).all():
        raise ValueError("native training tensors contain nonfinite values")
    diagnostic["max_abs_delta"] = float(np.max(np.abs(actual - reference)))
    if not np.allclose(actual, reference, rtol=0, atol=2e-6):
        raise ValueError(f"native training tensors disagree with slice-first reference: {diagnostic}")
    labels = dataset.labels
    if (labels is None or labels.dtype != torch.long or tuple(labels.shape) != (len(reference),)
            or bool(torch.any(labels != 0).item())):
        raise ValueError("null class labels must be all-zero long[N]")
    normalization = parsed.get("norm")
    fitted = dict(getattr(normalization, "kwargs", {}))
    for key in ("center", "xmax"):
        value = fitted.get(key)
        if value is None or not np.isfinite(value) or not np.isclose(value, metadata[key], rtol=0, atol=2e-6):
            raise ValueError(f"native fitted {key} disagrees with slice-first reference: {value} vs {metadata[key]}")
    return {"status": "matched", "contract": CONTRACT,
        "training_tensor_sha256": array_hash(actual),
        "training_reference_sha256": array_hash(reference),
        "selected_raw_sha256": metadata["selected_raw_sha256"],
        "reference_max_abs_delta": diagnostic["max_abs_delta"],
        "native_fitted_normalization": {key: float(fitted[key]) for key in ("center", "xmax")},
        "shape": diagnostic["shape"], "dtype": diagnostic["dtype"], "null_labels": "all_zero_long"}
