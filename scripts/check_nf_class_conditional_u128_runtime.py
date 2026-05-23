#!/usr/bin/env python
"""Runtime preflight for the u128 discrete class-conditional run.

This is meant to be run inside the same environment as the Slurm job before
the expensive training command.  It checks the fragile pieces that can fail
immediately on the cluster: torch/diffusers import compatibility, the external
cosmo_diffusion checkout, discrete class-label config, model forward support,
and a tiny data load with one simulation per field.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import yaml


DEFAULT_COSMODIFF_DIR = "/home/jiamingp/Diffusion_model/cosmo_diffusion_main"


def parse_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.replace("+", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


class TorchOptionalDeviceStub:
    """Small false backend so import-time optional-device probes do not fail."""

    def is_available(self) -> bool:
        return False

    def device_count(self) -> int:
        return 0

    def empty_cache(self) -> None:
        return None

    def _is_compiled(self) -> bool:
        return False

    def current_device(self) -> int:
        return 0

    def set_device(self, *args, **kwargs) -> None:
        return None

    def synchronize(self, *args, **kwargs) -> None:
        return None

    def manual_seed(self, *args, **kwargs) -> None:
        return None

    def manual_seed_all(self, *args, **kwargs) -> None:
        return None

    def seed(self, *args, **kwargs) -> int:
        return 0

    def initial_seed(self, *args, **kwargs) -> int:
        return 0

    def get_rng_state(self, *args, **kwargs):
        return None

    def set_rng_state(self, *args, **kwargs) -> None:
        return None

    def is_built(self, *args, **kwargs) -> bool:
        return False

    def current_stream(self, *args, **kwargs):
        return None

    def stream(self, *args, **kwargs):
        return nullcontext()

    def device(self, *args, **kwargs):
        return nullcontext()

    def memory_allocated(self, *args, **kwargs) -> int:
        return 0

    def max_memory_allocated(self, *args, **kwargs) -> int:
        return 0

    def reset_peak_memory_stats(self, *args, **kwargs) -> None:
        return None

    def get_device_name(self, *args, **kwargs) -> str:
        return "optional-device-unavailable"

    def get_device_properties(self, *args, **kwargs):
        return None

    def __getattr__(self, name: str):
        def missing(*args, **kwargs):
            if name.startswith("is_"):
                return False
            return None

        return missing


class TorchCompilerStub:
    """No-op torch.compiler facade for import-time diffusers decorators."""

    def disable(self, fn=None, recursive: bool = True):
        if fn is None:
            return lambda inner: inner
        return fn

    def is_compiling(self) -> bool:
        return False

    def is_exporting(self) -> bool:
        return False


def ensure_torch_optional_device_stubs():
    import torch

    stub = TorchOptionalDeviceStub()
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

    # diffusers>=0.32 imports TorchAO quantizer modules at model import time.
    # Those modules reference float8/float4 dtype names added in much newer
    # PyTorch releases.  The class-conditional UNet does not use quantization;
    # these aliases only keep the import path alive under torch 1.12.
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
    compiler_stub = TorchCompilerStub()
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
    if hasattr(torch, "distributed") and not hasattr(torch.distributed, "device_mesh"):
        torch.distributed.device_mesh = SimpleNamespace(DeviceMesh=object)
    if hasattr(torch, "distributed") and "torch.distributed._functional_collectives" not in sys.modules:
        funcol = ModuleType("torch.distributed._functional_collectives")

        class AsyncCollectiveTensor:
            pass

        def _identity_collective(tensor, *args, **kwargs):
            return tensor

        funcol.AsyncCollectiveTensor = AsyncCollectiveTensor
        funcol.all_to_all_single = _identity_collective
        funcol.all_gather_tensor = _identity_collective
        funcol.permute_tensor = _identity_collective
        sys.modules["torch.distributed._functional_collectives"] = funcol
        torch.distributed._functional_collectives = funcol
    return torch


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def preflight(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    cosmodiff_dir = Path(args.cosmodiff_dir).resolve()
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not (cosmodiff_dir / "cosmodiff" / "optim.py").exists():
        raise FileNotFoundError(f"Missing cosmodiff checkout: {cosmodiff_dir}")

    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, str(cosmodiff_dir))

    torch = ensure_torch_optional_device_stubs()
    print(f"[preflight] python={sys.executable}", flush=True)
    print(f"[preflight] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[preflight] torch_cuda_build={torch.version.cuda}", flush=True)
    print(f"[preflight] torch.xpu.is_available={torch.xpu.is_available()}", flush=True)
    print(f"[preflight] torch.mps.is_available={torch.mps.is_available()}", flush=True)
    print(f"[preflight] torch.float8_e4m3fn={torch.float8_e4m3fn}", flush=True)

    with config_path.open() as f:
        config = yaml.safe_load(f)
    requested_device = str(config.get("global", {}).get("device", ""))
    if requested_device.startswith("cuda") and torch.version.cuda is None:
        raise RuntimeError(
            "Config requests CUDA, but this environment is importing a CPU-only PyTorch build. "
            "Run scripts/setup_nf_class_conditional_env.sh after pulling the CUDA torch fix."
        )
    class_prefix = Path(sys.prefix).resolve()
    numpy_path = Path(np.__file__).resolve()
    print(f"[preflight] numpy={np.__version__} {numpy_path}", flush=True)
    if not numpy_path.is_relative_to(class_prefix):
        raise RuntimeError(
            f"numpy is being imported from {numpy_path}, outside the class env {class_prefix}. "
            "Run scripts/setup_nf_class_conditional_env.sh to install numpy/scipy into the class env."
        )
    try:
        import scipy
        import scipy.stats
    except Exception as exc:
        raise RuntimeError(
            "Failed to import scipy.stats. diffusers needs this for scheduler imports, including "
            "DPMSolverMultistepScheduler. Run scripts/setup_nf_class_conditional_env.sh so the "
            "class env shadows the incompatible Great Lakes Anaconda scipy build."
        ) from exc
    scipy_path = Path(scipy.__file__).resolve()
    print(f"[preflight] scipy={scipy.__version__} {scipy_path}", flush=True)
    if not scipy_path.is_relative_to(class_prefix):
        raise RuntimeError(
            f"scipy is being imported from {scipy_path}, outside the class env {class_prefix}. "
            "Run scripts/setup_nf_class_conditional_env.sh to install a compatible scipy wheel."
        )

    import diffusers
    from diffusers import AutoModel
    from cosmodiff import optim, utils

    torch_version = parse_version(torch.__version__)
    diffusers_version = parse_version(getattr(diffusers, "__version__", "0"))
    if torch_version < (2, 1) and diffusers_version >= (0, 32):
        print(
            "[preflight] torch<2.1 with recent diffusers; using Great Lakes optional-backend import shims",
            flush=True,
        )

    optim_path = Path(inspect.getsourcefile(optim)).resolve()
    utils_path = Path(inspect.getsourcefile(utils)).resolve()
    if cosmodiff_dir not in optim_path.parents:
        raise RuntimeError(f"Imported cosmodiff.optim from {optim_path}, expected under {cosmodiff_dir}")
    if cosmodiff_dir not in utils_path.parents:
        raise RuntimeError(f"Imported cosmodiff.utils from {utils_path}, expected under {cosmodiff_dir}")

    diffusers_path = Path(getattr(diffusers, "__file__", "unknown")).resolve()
    print(f"[preflight] diffusers={getattr(diffusers, '__version__', 'unknown')}", flush=True)
    print(f"[preflight] diffusers_file={diffusers_path}", flush=True)
    print(f"[preflight] diffusers.AutoModel={AutoModel}", flush=True)
    print(f"[preflight] cosmodiff.optim={optim_path}", flush=True)
    print(f"[preflight] cosmodiff.utils={utils_path}", flush=True)

    required_train = {
        "conditioning",
        "cfg_dropout",
        "ema_sigma_rels",
        "ema_update_every",
        "ema_burn_in",
        "min_snr_gamma",
        "sigma_log_normal",
    }
    missing = required_train.difference(inspect.signature(optim.train).parameters)
    if missing:
        raise RuntimeError(f"cosmodiff.optim.train missing required args: {sorted(missing)}")
    print("[preflight] cosmodiff train signature supports conditional training", flush=True)

    if config["train"].get("conditioning") != "discrete":
        raise RuntimeError(f"Expected train.conditioning=discrete, got {config['train'].get('conditioning')!r}")
    if config["generate"].get("conditioning") != "discrete":
        raise RuntimeError(f"Expected generate.conditioning=discrete, got {config['generate'].get('conditioning')!r}")
    if config["model"].get("class") != "UNet2DModel":
        raise RuntimeError(f"Expected UNet2DModel, got {config['model'].get('class')!r}")

    img_paths = [Path(p) for p in as_list(config["data"]["img_path"])]
    label_paths = [Path(p) for p in as_list(config["data"].get("label_path"))]
    if len(img_paths) != len(label_paths):
        raise RuntimeError(f"img_path and label_path counts differ: {len(img_paths)} vs {len(label_paths)}")

    missing_paths = [str(p) for p in [*img_paths, *label_paths] if not p.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing data/label files:\n" + "\n".join(missing_paths))

    label_values: list[int] = []
    for i, label_path in enumerate(label_paths):
        labels = np.load(label_path)
        if not np.issubdtype(labels.dtype, np.integer):
            raise RuntimeError(f"{label_path} must contain integer labels, got {labels.dtype}")
        if labels.ndim != 1:
            raise RuntimeError(f"{label_path} must be 1D, got shape {labels.shape}")
        uniq = sorted(int(x) for x in np.unique(labels).tolist())
        if uniq != [i]:
            raise RuntimeError(f"{label_path} expected only class id {i}, got {uniq}")
        label_values.extend(uniq)

    n_classes = len(img_paths)
    num_class_embeds = int(config["model"]["kwargs"].get("num_class_embeds", -1))
    if num_class_embeds < n_classes:
        raise RuntimeError(f"num_class_embeds={num_class_embeds} < n_classes={n_classes}")
    print(f"[preflight] class ids={label_values} num_class_embeds={num_class_embeds}", flush=True)

    if not args.skip_model_forward:
        model_kwargs = copy.deepcopy(config["model"]["kwargs"])
        model = getattr(diffusers, config["model"]["class"])(**model_kwargs).cpu().eval()
        sample_size = int(model_kwargs.get("sample_size", 128))
        in_channels = int(model_kwargs.get("in_channels", 1))
        labels = torch.arange(min(2, n_classes), dtype=torch.long)
        images = torch.zeros((labels.numel(), in_channels, sample_size, sample_size), dtype=torch.float32)
        timesteps = torch.zeros((labels.numel(),), dtype=torch.long)
        with torch.no_grad():
            out = model(images, timestep=timesteps, class_labels=labels, return_dict=False)[0]
        if tuple(out.shape) != tuple(images.shape):
            raise RuntimeError(f"Model forward shape mismatch: got {tuple(out.shape)}, expected {tuple(images.shape)}")
        print(f"[preflight] class-conditional model forward ok: output_shape={tuple(out.shape)}", flush=True)

    if not args.skip_small_data_load:
        small_config = copy.deepcopy(config)
        small_config.setdefault("global", {})["device"] = "cpu"
        small_config["data"]["keep_on_cpu"] = True
        small_config["data"]["n_samples"] = [1] * n_classes
        small_config["data"]["seed"] = [None] * n_classes
        data_out = utils.parse_config_data(small_config)
        dataset = data_out["data"] if isinstance(data_out, dict) else data_out
        sample = dataset[0]
        labels = sample.get("labels")
        if labels is None:
            raise RuntimeError("Small data load did not return labels.")
        if torch.is_floating_point(labels):
            raise RuntimeError(f"Discrete sample label should be integer, got {labels.dtype}")
        all_labels = dataset.labels.detach().cpu()
        uniq = sorted(int(x) for x in torch.unique(all_labels).tolist())
        if uniq != list(range(n_classes)):
            raise RuntimeError(f"Small data load expected class ids 0..{n_classes - 1}, got {uniq}")
        print(
            f"[preflight] small data load ok: len={len(dataset)} image_shape={tuple(sample['images'].shape)} "
            f"label_dtype={labels.dtype} class_ids={uniq}",
            flush=True,
        )

    print("[preflight] nf_class_conditional_u128 runtime checks passed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--cosmodiff-dir", default=DEFAULT_COSMODIFF_DIR)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-model-forward", action="store_true")
    parser.add_argument("--skip-small-data-load", action="store_true")
    return parser.parse_args()


def main() -> None:
    preflight(parse_args())


if __name__ == "__main__":
    main()
