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
    from types import ModuleType, SimpleNamespace

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

    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float4_e2m1fn_x2",
    ):
        if not hasattr(torch, name):
            setattr(torch, name, torch.float16)
    for bits in range(1, 8):
        name = f"uint{bits}"
        if not hasattr(torch, name):
            setattr(torch, name, torch.uint8)

    class _CompilerStub:
        def disable(self, fn=None, recursive=True):
            if fn is None:
                return lambda inner: inner
            return fn
        def is_compiling(self): return False
        def is_exporting(self): return False

    compiler_stub = _CompilerStub()
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        torch.compiler = compiler_stub
    else:
        for name in ("disable", "is_compiling", "is_exporting"):
            if not hasattr(compiler, name):
                setattr(compiler, name, getattr(compiler_stub, name))

    try:
        from torch.utils import _pytree
    except Exception:
        _pytree = None
    if _pytree is not None and not hasattr(_pytree, "register_pytree_node"):
        private_register = getattr(_pytree, "_register_pytree_node", None)
        if private_register is not None:
            def register_pytree_node(cls, flatten_fn, unflatten_fn, *args, **kwargs):
                try:
                    return private_register(cls, flatten_fn, unflatten_fn, *args, **kwargs)
                except TypeError:
                    supported = {
                        key: kwargs[key]
                        for key in ("to_dumpable_context", "from_dumpable_context")
                        if key in kwargs
                    }
                    try:
                        return private_register(cls, flatten_fn, unflatten_fn, *args, **supported)
                    except TypeError:
                        return private_register(cls, flatten_fn, unflatten_fn)
            _pytree.register_pytree_node = register_pytree_node

    try:
        import torch.distributed as torch_distributed
    except Exception:
        torch_distributed = None
    if torch_distributed is not None and not hasattr(torch_distributed, "device_mesh"):
        torch_distributed.device_mesh = SimpleNamespace(DeviceMesh=object)
    if torch_distributed is not None and "torch.distributed._functional_collectives" not in sys.modules:
        funcol = ModuleType("torch.distributed._functional_collectives")

        class AsyncCollectiveTensor:
            pass

        def identity_collective(tensor, *args, **kwargs):
            return tensor

        funcol.AsyncCollectiveTensor = AsyncCollectiveTensor
        funcol.all_to_all_single = identity_collective
        funcol.all_gather_tensor = identity_collective
        funcol.permute_tensor = identity_collective
        sys.modules["torch.distributed._functional_collectives"] = funcol
        torch_distributed._functional_collectives = funcol


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


def _load_dit_direct(checkpoint: Path, config_path: Path | None, *, allow_default_scheduler: bool = False):
    """Load DiT checkpoints directly for transformer sampling."""
    from diffusers import DiTTransformer2DModel

    model = DiTTransformer2DModel.from_pretrained(str(checkpoint))
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

    if model_class in {"DiTTransformer2DModel", "diffusers.DiTTransformer2DModel"}:
        try:
            return _load_dit_direct(
                checkpoint,
                config_path,
                allow_default_scheduler=allow_default_scheduler,
            )
        except Exception as exc:
            print(f"Direct DiT load failed, trying cosmodiff load_checkpoint: {exc}")

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
            if model_class in {"DiTTransformer2DModel", "diffusers.DiTTransformer2DModel"}:
                return _load_dit_direct(
                    checkpoint,
                    config_path,
                    allow_default_scheduler=allow_default_scheduler,
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


def _checkpoint_root_for_ema(requested_checkpoint: Path, resolved_checkpoint: Path) -> Path:
    if requested_checkpoint.is_dir() and not _looks_like_checkpoint(requested_checkpoint):
        return requested_checkpoint
    if resolved_checkpoint.name.startswith("checkpoint-epoch-"):
        return resolved_checkpoint.parent
    return requested_checkpoint.parent if requested_checkpoint.is_dir() else resolved_checkpoint.parent


def _apply_posthoc_ema(
    model: torch.nn.Module,
    *,
    requested_checkpoint: Path,
    resolved_checkpoint: Path,
    sigma_rel: float | None,
) -> torch.nn.Module:
    if sigma_rel is None:
        return model

    try:
        from cosmodiff.optim import synthesize_ema_from_checkpoints
    except ImportError as exc:
        raise RuntimeError(
            "--ema-sigma-rel requires cosmodiff.optim.synthesize_ema_from_checkpoints "
            "from the patched Great Lakes cosmo_diffusion checkout."
        ) from exc

    checkpoint_root = _checkpoint_root_for_ema(requested_checkpoint, resolved_checkpoint)
    ema_model = synthesize_ema_from_checkpoints(
        model,
        str(checkpoint_root),
        sigma_rel_target=float(sigma_rel),
    )
    return ema_model.ema_model if hasattr(ema_model, "ema_model") else ema_model


def generate_samples(
    model: torch.nn.Module,
    noise_scheduler,
    *,
    batch_size: int,
    image_shape: tuple[int, ...],
    num_steps: int | None,
    device: torch.device,
    generator: torch.Generator | None,
    class_labels: torch.Tensor | None = None,
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
        if class_labels is not None:
            labels = class_labels.to(device=device, dtype=torch.long)
            noise_pred = model(
                images,
                timestep=timesteps,
                class_labels=labels,
                return_dict=False,
            )[0]
        else:
            noise_pred = model(images, timesteps, return_dict=False)[0]
        step_kwargs = {}
        if "generator" in step_params:
            step_kwargs["generator"] = generator
        images = noise_scheduler.step(noise_pred, t, images, **step_kwargs).prev_sample

    return images


def _labels_for_batch(
    *,
    labels: np.ndarray | None,
    class_label: int | None,
    start: int,
    batch_size: int,
) -> torch.Tensor | None:
    if labels is not None:
        end = start + batch_size
        if end > len(labels):
            raise ValueError(f"Requested labels[{start}:{end}] but only {len(labels)} labels are available.")
        return torch.as_tensor(labels[start:end], dtype=torch.long)
    if class_label is not None:
        return torch.full((batch_size,), int(class_label), dtype=torch.long)
    return None


def save_sample_output(
    output: Path,
    samples: np.ndarray,
    *,
    requested_checkpoint: Path,
    resolved_checkpoint: Path,
    config_path: Path | None,
    scheduler_name: str,
    num_steps: int,
    seed: int,
) -> None:
    """Save samples with enough provenance to audit checkpoint comparisons."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".npz":
        np.savez(
            output,
            samples=samples,
            requested_checkpoint=np.asarray(str(requested_checkpoint)),
            resolved_checkpoint=np.asarray(str(resolved_checkpoint)),
            config_path=np.asarray(str(config_path) if config_path is not None else ""),
            scheduler=np.asarray(str(scheduler_name)),
            num_steps=np.asarray(int(num_steps), dtype=np.int64),
            seed=np.asarray(int(seed), dtype=np.int64),
        )
    else:
        np.save(output, samples)


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
    parser.add_argument("--class-label", type=int, default=None, help="Use one class label for every generated sample.")
    parser.add_argument("--labels", default=None, help="Optional .npy file with one integer class label per generated sample.")
    parser.add_argument(
        "--ema-sigma-rel",
        type=float,
        default=None,
        help="Optional post-hoc EMA target sigma_rel synthesized from the checkpoint root.",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Load the model/scheduler and exit without sampling.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _reject_known_bad_runtime()
    project_root = Path.cwd()
    _install_torch_optional_device_stubs()
    _install_sklearn_roc_curve_stub()
    _ensure_cosmodiff_on_path(project_root)

    requested_checkpoint = Path(args.checkpoint)
    checkpoint = requested_checkpoint
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
    model = _apply_posthoc_ema(
        model,
        requested_checkpoint=requested_checkpoint,
        resolved_checkpoint=checkpoint,
        sigma_rel=args.ema_sigma_rel,
    )
    scheduler = build_inference_scheduler(scheduler, args.scheduler)
    n_steps = int(args.num_steps or scheduler.config.num_train_timesteps)
    if args.preflight_only:
        scheduler.set_timesteps(n_steps)
        print(
            "preflight ok: "
            f"checkpoint={checkpoint} scheduler={scheduler.__class__.__name__} "
            f"steps={len(scheduler.timesteps)} ema_sigma_rel={args.ema_sigma_rel}"
        )
        return

    print(f"resolved checkpoint: {checkpoint}")

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    batches = []
    remaining = args.num_samples
    labels = np.load(args.labels) if args.labels else None
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(labels) < args.num_samples:
            raise ValueError(f"--labels has {len(labels)} labels, but --num-samples={args.num_samples}.")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    offset = 0
    with torch.no_grad():
        while remaining > 0:
            n = min(args.batch_size, remaining)
            batch_labels = _labels_for_batch(
                labels=labels,
                class_label=args.class_label,
                start=offset,
                batch_size=n,
            )
            samples = generate_samples(
                model,
                scheduler,
                batch_size=n,
                image_shape=(1, args.image_size, args.image_size),
                num_steps=args.num_steps,
                device=device,
                generator=generator,
                class_labels=batch_labels,
            )
            batches.append(samples.detach().cpu().numpy())
            remaining -= n
            offset += n

    output = Path(args.output)
    samples = np.concatenate(batches, axis=0)
    save_sample_output(
        output,
        samples,
        requested_checkpoint=requested_checkpoint,
        resolved_checkpoint=checkpoint,
        config_path=config_path,
        scheduler_name=scheduler.__class__.__name__,
        num_steps=n_steps,
        seed=args.seed,
    )
    print(f"Wrote {args.num_samples} samples to {output}")


if __name__ == "__main__":
    main()
