#!/usr/bin/env python
"""Generate samples from a cosmodiff checkpoint and save them as ``.npy``."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = re.findall(r"\d+", version.split("+", 1)[0])
    padded = (parts + ["0", "0", "0"])[:3]
    return tuple(int(part) for part in padded)


def _reject_known_bad_runtime() -> None:
    try:
        diffusers_version = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return

    if _version_tuple(torch.__version__) < (2, 1, 0) and _version_tuple(diffusers_version) >= (0, 38, 0):
        raise SystemExit(
            "This Python environment has torch "
            f"{torch.__version__} with diffusers {diffusers_version}, which is not a usable "
            "Great Lakes sampling runtime. Use /home/jiamingp/venvs/cosmodiff_nf_class "
            "or run scripts/setup_nf_class_conditional_env.sh first."
        )


def _install_torch_optional_device_stubs() -> None:
    """Mask optional accelerator APIs that old Torch builds do not expose.

    Newer diffusers versions reference ``torch.xpu`` while importing, even on
    CPU/CUDA-only systems.  Older Great Lakes Torch builds do not have that
    attribute, so declare it unavailable before importing diffusers.
    """
    from contextlib import nullcontext

    class _OptionalDeviceStub:
        def is_available(self): return False
        def device_count(self): return 0
        def empty_cache(self): return None
        def _is_compiled(self): return False
        def current_device(self): return 0
        def set_device(self, *args, **kwargs): return None
        def synchronize(self, *args, **kwargs): return None
        def manual_seed(self, *args, **kwargs): return None
        def manual_seed_all(self, *args, **kwargs): return None
        def seed(self, *args, **kwargs): return 0
        def initial_seed(self, *args, **kwargs): return 0
        def get_rng_state(self, *args, **kwargs): return None
        def set_rng_state(self, *args, **kwargs): return None
        def is_built(self, *args, **kwargs): return False
        def current_stream(self, *args, **kwargs): return None
        def stream(self, *args, **kwargs): return nullcontext()
        def device(self, *args, **kwargs): return nullcontext()
        def memory_allocated(self, *args, **kwargs): return 0
        def max_memory_allocated(self, *args, **kwargs): return 0
        def reset_peak_memory_stats(self, *args, **kwargs): return None
        def get_device_name(self, *args, **kwargs): return "optional-device-unavailable"
        def get_device_properties(self, *args, **kwargs): return None
        def __getattr__(self, name):
            def missing(*args, **kwargs):
                if name.startswith("is_"):
                    return False
                return None
            return missing

    stub = _OptionalDeviceStub()
    required = ("empty_cache", "is_available", "device_count", "manual_seed")
    for backend in ("xpu", "mps"):
        existing = getattr(torch, backend, None)
        if existing is None or any(not hasattr(existing, name) for name in required):
            setattr(torch, backend, stub)
            continue
        for name in dir(stub):
            if name.startswith("__"):
                continue
            if not hasattr(existing, name):
                setattr(existing, name, getattr(stub, name))


def _install_sklearn_roc_curve_stub() -> None:
    """Avoid optional transformers -> sklearn imports on mixed HPC envs.

    Recent diffusers/transformers can import ``sklearn.metrics.roc_curve`` while
    importing UNet classes, even though diffusion sampling does not need it.  On
    Great Lakes the Anaconda sklearn binary can fail with a GLIBCXX error.  A
    minimal stub keeps the optional import path from touching that binary.
    """
    import os
    from importlib.machinery import ModuleSpec
    import types

    if os.environ.get("COSMODIFF_DISABLE_SKLEARN_STUB") == "1":
        return
    if "sklearn.metrics" in sys.modules:
        return

    sklearn = types.ModuleType("sklearn")
    metrics = types.ModuleType("sklearn.metrics")
    sklearn.__spec__ = ModuleSpec("sklearn", loader=None, is_package=True)
    sklearn.__path__ = []
    metrics.__spec__ = ModuleSpec("sklearn.metrics", loader=None, is_package=True)
    metrics.__path__ = []

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
        if not path.exists():
            raise FileNotFoundError(f"COSMODIFF_DIR does not exist: {path}")
        if str(path) not in sys.path:
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


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-epoch-*"):
        if not path.is_dir():
            continue
        try:
            epoch = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoints.append((epoch, path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


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


def build_inference_scheduler(base_scheduler, scheduler_name: str | None):
    """Optionally replace the training scheduler with an inference scheduler."""
    if not scheduler_name:
        return base_scheduler

    import diffusers

    scheduler_cls = getattr(diffusers, scheduler_name)
    if hasattr(scheduler_cls, "from_config"):
        return scheduler_cls.from_config(base_scheduler.config)
    return scheduler_cls(**dict(base_scheduler.config))


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

    from cosmodiff import utils

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


def generate_samples(
    model: torch.nn.Module,
    noise_scheduler,
    *,
    batch_size: int,
    image_shape: tuple[int, ...],
    num_steps: int | None,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    model.eval()
    n_steps = int(num_steps or noise_scheduler.config.num_train_timesteps)
    noise_scheduler.set_timesteps(n_steps)

    images = torch.randn((batch_size, *image_shape), device=device, generator=generator)

    try:
        step_params = inspect.signature(noise_scheduler.step).parameters
    except (TypeError, ValueError):
        step_params = {}

    for t in noise_scheduler.timesteps:
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = model(images, timesteps, return_dict=False)[0]
        step_kwargs = {}
        if "generator" in step_params:
            step_kwargs["generator"] = generator
        images = noise_scheduler.step(noise_pred, t, images, **step_kwargs).prev_sample

    return images


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
    parser.add_argument("--scheduler", default=None, help="Optional inference scheduler class, e.g. DPMSolverMultistepScheduler.")
    parser.add_argument("--num-steps", type=int, default=None, help="Optional inference-step count for the scheduler.")
    parser.add_argument("--preflight-only", action="store_true", help="Load the model/scheduler and exit without sampling.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _reject_known_bad_runtime()
    project_root = Path.cwd()
    _install_torch_optional_device_stubs()
    _install_sklearn_roc_curve_stub()
    _ensure_cosmodiff_on_path(project_root)

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir() and not _looks_like_checkpoint(checkpoint):
        latest = _find_latest_checkpoint(checkpoint)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found under {checkpoint}")
        checkpoint = latest

    config_path = Path(args.config) if args.config else None
    model, scheduler = _load_for_sampling(
        checkpoint,
        config_path,
        allow_default_scheduler=args.allow_default_scheduler,
    )
    scheduler = build_inference_scheduler(scheduler, args.scheduler)
    if args.preflight_only:
        n_steps = int(args.num_steps or scheduler.config.num_train_timesteps)
        scheduler.set_timesteps(n_steps)
        print(
            "preflight ok: "
            f"checkpoint={checkpoint} scheduler={scheduler.__class__.__name__} steps={len(scheduler.timesteps)}"
        )
        return

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    batches = []
    remaining = args.num_samples
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        while remaining > 0:
            n = min(args.batch_size, remaining)
            samples = generate_samples(
                model,
                scheduler,
                batch_size=n,
                image_shape=(1, args.image_size, args.image_size),
                num_steps=args.num_steps,
                device=device,
                generator=generator,
            )
            batches.append(samples.detach().cpu().numpy())
            remaining -= n

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = np.concatenate(batches, axis=0)
    if output.suffix == ".npz":
        np.savez(output, samples=samples)
    else:
        np.save(output, samples)
    print(f"Wrote {args.num_samples} samples to {output}")


if __name__ == "__main__":
    main()
