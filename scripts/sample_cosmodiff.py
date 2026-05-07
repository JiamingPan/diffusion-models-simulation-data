#!/usr/bin/env python
"""Generate samples from a cosmodiff checkpoint and save them as ``.npy``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _ensure_cosmodiff_on_path(project_root: Path) -> None:
    candidate = project_root / "cosmo_diffusion"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _looks_like_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (
            (path / "config.json").exists()
            or (path / "model_index.json").exists()
            or any(path.glob("diffusion_pytorch_model.*"))
            or path.name.startswith("checkpoint-")
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or run output directory.")
    parser.add_argument("--output", required=True, help="Output .npy path.")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    project_root = Path.cwd()
    _ensure_cosmodiff_on_path(project_root)
    from cosmodiff import utils
    from cosmodiff.optim import generate

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir() and not _looks_like_checkpoint(checkpoint):
        latest = utils.find_latest_checkpoint(str(checkpoint))
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found under {checkpoint}")
        checkpoint = Path(latest)

    model, scheduler, _, _, _ = utils.load_checkpoint(str(checkpoint))
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    batches = []
    remaining = args.num_samples
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        while remaining > 0:
            n = min(args.batch_size, remaining)
            samples = generate(
                model,
                scheduler,
                batch_size=n,
                image_shape=(1, args.image_size, args.image_size),
                device=device,
                generator=generator,
            )
            batches.append(samples.detach().cpu().numpy())
            remaining -= n

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.concatenate(batches, axis=0))
    print(f"Wrote {args.num_samples} samples to {output}")


if __name__ == "__main__":
    main()
