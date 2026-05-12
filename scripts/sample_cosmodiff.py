#!/usr/bin/env python
"""Generate samples from a cosmodiff checkpoint and save them as ``.npy``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def _install_sklearn_roc_curve_stub() -> None:
    """Avoid optional transformers -> sklearn imports on mixed HPC envs.

    Recent diffusers/transformers can import ``sklearn.metrics.roc_curve`` while
    importing UNet classes, even though diffusion sampling does not need it.  On
    Great Lakes the Anaconda sklearn binary can fail with a GLIBCXX error.  A
    minimal stub keeps the optional import path from touching that binary.
    """
    import os
    import types

    if os.environ.get("COSMODIFF_DISABLE_SKLEARN_STUB") == "1":
        return
    if "sklearn.metrics" in sys.modules:
        return

    sklearn = types.ModuleType("sklearn")
    metrics = types.ModuleType("sklearn.metrics")

    def roc_curve(*_args, **_kwargs):
        raise RuntimeError("sklearn.metrics.roc_curve is stubbed for cosmodiff sampling.")

    metrics.roc_curve = roc_curve
    sklearn.metrics = metrics
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.metrics", metrics)


def _ensure_cosmodiff_on_path(project_root: Path) -> None:
    import importlib.util
    import os

    env_candidate = os.environ.get("COSMODIFF_DIR")
    if env_candidate:
        path = Path(env_candidate)
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
        return

    if importlib.util.find_spec("cosmodiff") is not None:
        return

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


def _load_scheduler_from_config(config_path: Path | None, *, allow_default_scheduler: bool = False):
    import diffusers
    from diffusers import DDPMScheduler

    if config_path is None:
        if not allow_default_scheduler:
            raise ValueError(
                "Checkpoint is missing saved scheduler metadata, so --config is required "
                "to reconstruct the training scheduler. Pass --allow-default-scheduler "
                "only for explicit smoke tests."
            )
        print("No --config supplied; using DDPMScheduler(num_train_timesteps=1000).")
        return DDPMScheduler(num_train_timesteps=1000)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    scheduler_config = config.get("noise_scheduler")
    if scheduler_config is None:
        print(f"{config_path} has no noise_scheduler block; using DDPMScheduler(num_train_timesteps=1000).")
        return DDPMScheduler(num_train_timesteps=1000)

    scheduler_cls = getattr(diffusers, scheduler_config["class"])
    return scheduler_cls(**scheduler_config.get("kwargs", {}))


def _config_model_class(config_path: Path | None) -> str | None:
    if config_path is None:
        return None
    with config_path.open() as f:
        config = yaml.safe_load(f)
    return config.get("model", {}).get("class")


def _load_unet_direct(checkpoint: Path, config_path: Path | None, *, allow_default_scheduler: bool = False):
    """Load UNet checkpoints without diffusers.AutoModel.

    Some Great Lakes environments leak an old user-site ``transformers`` into
    the venv.  ``diffusers.AutoModel`` then imports optional autoencoder code
    and fails before it reaches the UNet.  Direct UNet loading avoids that
    unrelated import path.
    """
    from diffusers import UNet2DModel

    model = UNet2DModel.from_pretrained(str(checkpoint))
    scheduler = _load_scheduler_from_config(
        config_path,
        allow_default_scheduler=allow_default_scheduler,
    )
    return model, scheduler


def _load_for_sampling(checkpoint: Path, config_path: Path | None, *, allow_default_scheduler: bool = False):
    from cosmodiff import utils

    model_class = _config_model_class(config_path)
    if model_class in {"UNet2DModel", "diffusers.UNet2DModel"}:
        try:
            return _load_unet_direct(
                checkpoint,
                config_path,
                allow_default_scheduler=allow_default_scheduler,
            )
        except Exception as exc:
            print(f"Direct UNet load failed, trying cosmodiff load_checkpoint: {exc}")

    try:
        model, scheduler, _, _, _ = utils.load_checkpoint(str(checkpoint))
        return model, scheduler
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        if not isinstance(exc, FileNotFoundError):
            print(
                "cosmodiff load_checkpoint failed; loading UNet weights directly "
                f"and reconstructing the scheduler from config. Error: {exc}"
            )
            return _load_unet_direct(
                checkpoint,
                config_path,
                allow_default_scheduler=allow_default_scheduler,
            )
        missing = Path(exc.filename or "")
        if missing.name not in {"checkpoint_config.yaml", "noise_scheduler.pkl", "optimizer.pkl", "lr_scheduler.pkl"}:
            raise
        print(
            f"{checkpoint} is missing {missing.name}; loading UNet weights directly "
            "and reconstructing the noise scheduler."
        )

    return _load_unet_direct(
        checkpoint,
        config_path,
        allow_default_scheduler=allow_default_scheduler,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or run output directory.")
    parser.add_argument("--config", default=None, help="Optional run YAML used to reconstruct the noise scheduler.")
    parser.add_argument(
        "--allow-default-scheduler",
        action="store_true",
        help="Allow fallback to DDPMScheduler(num_train_timesteps=1000) when checkpoint scheduler metadata is missing.",
    )
    parser.add_argument("--output", required=True, help="Output .npy path.")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    project_root = Path.cwd()
    _install_sklearn_roc_curve_stub()
    _ensure_cosmodiff_on_path(project_root)
    from cosmodiff.optim import generate
    from cosmodiff import utils

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir() and not _looks_like_checkpoint(checkpoint):
        latest = utils.find_latest_checkpoint(str(checkpoint))
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found under {checkpoint}")
        checkpoint = Path(latest)

    config_path = Path(args.config) if args.config else None
    model, scheduler = _load_for_sampling(
        checkpoint,
        config_path,
        allow_default_scheduler=args.allow_default_scheduler,
    )
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
