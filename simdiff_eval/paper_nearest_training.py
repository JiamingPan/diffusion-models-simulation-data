"""Fail-closed data products for the paper nearest-training figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from simdiff_eval.io import (
    as_nchw,
    configured_training_reference_info,
    iter_real_reference_batches_from_config,
)
from simdiff_eval.metrics import (
    batch_power_spectra,
    frechet_feature_distance,
    real_split_frechet_baseline,
)


def load_npz_samples(path: str | Path) -> np.ndarray:
    """Load generated samples from the repository's supported NPZ layouts."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing generated sample archive: {source}")
    with np.load(source) as payload:
        if "samples" in payload.files:
            array = payload["samples"]
        elif "arr_0" in payload.files:
            array = payload["arr_0"]
        elif payload.files:
            array = payload[payload.files[0]]
        else:
            raise ValueError(f"generated sample archive has no arrays: {source}")
    return as_nchw(np.asarray(array, dtype=np.float32))


def summarize_exact_training_reference(
    config_path: str | Path,
    *,
    expected_slices: int,
    generated: np.ndarray,
    nbins: int = 30,
    k_max: float = 64.0,
) -> dict[str, Any]:
    """Compare generated fields with the complete configured training subset.

    The reference is streamed so even the largest selected training subset does
    not need to be materialized at once. Both the config-derived and streamed
    slice counts must agree with the manifest-provided ``expected_slices``.
    """

    config_path = Path(config_path)
    reference_info = configured_training_reference_info(config_path)
    configured_slices = int(reference_info["configured_slices"])
    expected_slices = int(expected_slices)
    if configured_slices != expected_slices:
        raise RuntimeError(
            "manifest/config slice-count mismatch: "
            f"manifest={expected_slices}, config={configured_slices}, config_path={config_path}"
        )

    generated_arr = as_nchw(np.asarray(generated, dtype=np.float32))
    if len(generated_arr) < 1:
        raise ValueError("generated sample set is empty")
    generated_pk, generated_k = batch_power_spectra(generated_arr, nbins=int(nbins))
    generated_finite = np.isfinite(generated_pk)
    generated_mean = np.nansum(generated_pk, axis=0) / np.maximum(
        generated_finite.sum(axis=0),
        1,
    )
    generated_mean[generated_finite.sum(axis=0) == 0] = np.nan

    query = generated_arr[0].reshape(-1).astype(np.float32, copy=False)
    query_norm = float(np.linalg.norm(query))
    nearest_mse = float("inf")
    nearest_cosine = float("nan")
    nearest_index = -1
    nearest_image: np.ndarray | None = None
    real_pk_sum: np.ndarray | None = None
    real_pk_count = np.zeros_like(generated_mean, dtype=np.int64)
    loaded_slices = 0
    for batch in iter_real_reference_batches_from_config(config_path):
        reference = as_nchw(np.asarray(batch, dtype=np.float32))
        batch_pk, batch_k = batch_power_spectra(reference, nbins=int(nbins))
        if not np.allclose(batch_k, generated_k, equal_nan=True):
            raise RuntimeError(f"power-spectrum bin mismatch for {config_path}")
        finite = np.isfinite(batch_pk)
        batch_sum = np.nansum(batch_pk, axis=0)
        real_pk_sum = batch_sum if real_pk_sum is None else real_pk_sum + batch_sum
        real_pk_count += finite.sum(axis=0)

        flat = reference.reshape(len(reference), -1)
        difference = flat - query[None, :]
        mse = np.mean(difference * difference, axis=1)
        local = int(np.argmin(mse))
        if float(mse[local]) < nearest_mse:
            nearest_mse = float(mse[local])
            nearest_index = loaded_slices + local
            nearest_image = reference[local, 0].copy()
            denominator = query_norm * float(np.linalg.norm(flat[local]))
            nearest_cosine = (
                float(np.dot(query, flat[local]) / denominator) if denominator > 0.0 else 0.0
            )
        loaded_slices += len(reference)

    if loaded_slices != configured_slices:
        raise RuntimeError(
            "streamed/config slice-count mismatch: "
            f"streamed={loaded_slices}, config={configured_slices}, config_path={config_path}"
        )
    if real_pk_sum is None or nearest_image is None or nearest_index < 0:
        raise RuntimeError(f"no exact-subset reference fields were loaded from {config_path}")

    real_mean = real_pk_sum / np.maximum(real_pk_count, 1)
    valid = (
        np.isfinite(generated_k)
        & (generated_k <= float(k_max))
        & np.isfinite(generated_mean)
        & np.isfinite(real_mean)
        & (real_mean > 0.0)
    )
    if not np.any(valid):
        raise RuntimeError(f"no finite power-spectrum bins at k <= {k_max:g} for {config_path}")
    k_bins = generated_k[valid]
    ratio = generated_mean[valid] / real_mean[valid]
    pk_log10_mae = float(np.mean(np.abs(np.log10(np.clip(ratio, 1e-30, None)))))
    return {
        "n_training_slices": loaded_slices,
        "n_generated": int(len(generated_arr)),
        "nearest_training_index": nearest_index,
        "nearest_training_image": nearest_image,
        "nearest_mse": nearest_mse,
        "nearest_cosine": nearest_cosine,
        "k_bins": k_bins,
        "pk_ratio": ratio,
        "pk_log10_mae": pk_log10_mae,
        "k_max": float(k_max),
        "reference_info": reference_info,
    }


def resolve_sscd_embedding_cache(
    cache_dir: str | Path,
    *,
    run_name: str,
    kind: str,
    sample_label: str,
    seed: int,
) -> Path:
    """Resolve the latest matching full-NN SSCD cache or fail closed."""

    root = Path(cache_dir)
    pattern = f"{run_name}_{kind}_{sample_label}_seed{int(seed)}_*.pt"
    matches = sorted(
        root.glob(pattern),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not matches:
        raise FileNotFoundError(f"missing SSCD embedding cache: {root / pattern}")
    return matches[-1]


def load_sscd_embedding_cache(path: str | Path) -> np.ndarray:
    """Load a two-dimensional embedding matrix from a cached Torch payload."""

    import torch

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing SSCD embedding cache: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    tensor = payload["embeddings"] if isinstance(payload, dict) else payload
    features = np.asarray(tensor.detach().cpu(), dtype=np.float64)
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError(f"invalid SSCD embedding cache {source}: shape={features.shape}")
    return features


def _project_to_real_pca(
    heldout: np.ndarray,
    generated: np.ndarray,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    center = heldout.mean(axis=0, keepdims=True)
    centered_real = heldout - center
    centered_generated = generated - center
    rank = min(int(max_components), centered_real.shape[0] - 2, centered_real.shape[1])
    if rank < 1:
        raise ValueError("not enough heldout real embeddings for PCA projection")
    _, _, right_vectors = np.linalg.svd(centered_real, full_matrices=False)
    basis = right_vectors[:rank].T
    return centered_real @ basis, centered_generated @ basis, rank


def normalized_sscd_frechet(
    heldout_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    seed: int = 123,
    max_components: int = 64,
) -> dict[str, float | int]:
    """Reuse the established SSCD/PCA Fréchet ratio with equal-sized splits."""

    heldout = np.asarray(heldout_features, dtype=np.float64)
    generated = np.asarray(generated_features, dtype=np.float64)
    if heldout.ndim != 2 or generated.ndim != 2:
        raise ValueError("SSCD embeddings must be two-dimensional")
    if heldout.shape[1] != generated.shape[1]:
        raise ValueError("heldout and generated SSCD feature dimensions differ")
    n_available_generated = len(generated)
    if n_available_generated < 2:
        raise ValueError("generated SSCD cache must contain at least two samples")
    n_eval = min(n_available_generated, len(heldout) // 2)
    if n_eval < 2:
        raise ValueError(
            "heldout SSCD cache cannot provide two equal real splits: "
            f"heldout={len(heldout)}, generated={n_available_generated}"
        )

    heldout_projected, generated_projected, rank = _project_to_real_pca(
        heldout,
        generated,
        max_components,
    )
    rng = np.random.default_rng(int(seed))
    real_indices = rng.permutation(len(heldout_projected))[: 2 * n_eval]
    generated_indices = rng.permutation(len(generated_projected))[:n_eval]
    real_first = heldout_projected[real_indices[:n_eval]]
    real_second = heldout_projected[real_indices[n_eval:]]
    generated_eval = generated_projected[generated_indices]

    baseline = real_split_frechet_baseline(
        np.concatenate((real_first, real_second), axis=0),
        seed=int(seed),
    )
    baseline_distance = float(baseline["distance"])
    generated_distance = frechet_feature_distance(generated_eval, real_first)
    return {
        "sscd_frechet_normalized": float(generated_distance / max(baseline_distance, 1e-12)),
        "generated_to_real_frechet": float(generated_distance),
        "real_split_frechet": baseline_distance,
        "n_generated": int(n_eval),
        "n_real_split": int(baseline["n_first"]),
        "pca_rank": int(rank),
        "seed": int(seed),
    }
