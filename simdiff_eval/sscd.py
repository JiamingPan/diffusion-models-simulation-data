"""SSCD helpers for paper-style copy/generalization diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


RenderMode = Literal["fixed", "per_image"]


def load_sscd_torchscript(path: str | Path, device: str | torch.device | None = None) -> torch.nn.Module:
    """Load a TorchScript SSCD model.

    The expected model is usually ``sscd_disc_mixup.torchscript.pt`` from
    facebookresearch/sscd-copy-detection.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = torch.jit.load(str(path), map_location=device)
    model.to(device)
    model.eval()
    return model


def _as_nchw_tensor(images: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(images, np.ndarray) and not images.flags.writeable:
        images = images.copy()
    tensor = torch.as_tensor(images, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor[:, None, :, :]
    if tensor.ndim != 4 or tensor.shape[1] not in (1, 3):
        raise ValueError(f"Expected images shaped (N,H,W), (N,1,H,W), or (N,3,H,W); got {tuple(tensor.shape)}.")
    return tensor


def fields_to_sscd_input(
    images: np.ndarray | torch.Tensor,
    *,
    image_size: int = 320,
    render_mode: RenderMode = "fixed",
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Convert scalar CAMELS fields to SSCD-ready RGB tensors.

    SSCD was trained on RGB natural images. For scalar fields, we make a
    deterministic grayscale rendering, resize to a square tensor, and apply
    ImageNet normalization, matching the public SSCD inference recipe.

    ``render_mode="fixed"`` maps a shared value range, usually ``[-1, 1]``, to
    ``[0, 1]``. This preserves amplitude differences between real/generated
    fields. ``render_mode="per_image"`` rescales each image independently and
    emphasizes morphology over absolute amplitude.
    """
    tensor = _as_nchw_tensor(images)

    if render_mode == "fixed":
        lo, hi = value_range
        if hi <= lo:
            raise ValueError("value_range must satisfy hi > lo.")
        tensor = (tensor - lo) / (hi - lo)
        tensor = tensor.clamp(0.0, 1.0)
    elif render_mode == "per_image":
        flat = tensor.flatten(1)
        lo = flat.min(dim=1).values[:, None, None, None]
        hi = flat.max(dim=1).values[:, None, None, None]
        tensor = (tensor - lo) / (hi - lo).clamp_min(1e-12)
        tensor = tensor.clamp(0.0, 1.0)
    else:
        raise ValueError("render_mode must be 'fixed' or 'per_image'.")

    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device)[None, :, None, None]
    return (tensor - mean) / std


@torch.no_grad()
def sscd_embeddings(
    images: np.ndarray | torch.Tensor,
    model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
    image_size: int = 320,
    render_mode: RenderMode = "fixed",
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Embed scalar fields with SSCD and return L2-normalized CPU features."""
    device = torch.device(device or next(model.parameters()).device)
    features: list[torch.Tensor] = []

    n = len(images)
    for start in range(0, n, batch_size):
        batch = fields_to_sscd_input(
            images[start:start + batch_size],
            image_size=image_size,
            render_mode=render_mode,
            value_range=value_range,
        ).to(device)
        emb = model(batch)
        emb = F.normalize(emb, dim=1)
        features.append(emb.cpu())

    return torch.cat(features, dim=0)


@torch.no_grad()
def sscd_generalization_metrics(
    generated_embeddings: torch.Tensor,
    training_embeddings: torch.Tensor,
    *,
    threshold: float = 0.6,
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute paper-style SSCD generalizability metrics.

    For each generated sample ``x``, compute its maximum cosine similarity to
    any training sample ``y_i``. A generated sample is counted as a near-copy if
    ``max_i sim(x, y_i) > threshold``.

    ``generalization_score = 1 - copy_fraction``.
    """
    generated_embeddings = F.normalize(generated_embeddings.float(), dim=1)
    training_embeddings = F.normalize(training_embeddings.float(), dim=1)

    max_sims = []
    for start in range(0, len(generated_embeddings), batch_size):
        gen = generated_embeddings[start:start + batch_size]
        sim = gen @ training_embeddings.T
        max_sims.append(sim.max(dim=1).values.cpu())

    max_sim = torch.cat(max_sims)
    copy_mask = max_sim > threshold

    return {
        "generalization_score": float((~copy_mask).float().mean().item()),
        "copy_fraction": float(copy_mask.float().mean().item()),
        "max_similarity_mean": float(max_sim.mean().item()),
        "max_similarity_median": float(max_sim.median().item()),
        "max_similarity_q90": float(torch.quantile(max_sim, 0.90).item()),
        "max_similarity_q99": float(torch.quantile(max_sim, 0.99).item()),
        "max_similarity_max": float(max_sim.max().item()),
        "threshold": float(threshold),
    }
