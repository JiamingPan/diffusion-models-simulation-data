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
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.torch_compat import install_torch_backend_compat


DEFAULT_COSMODIFF_DIR = "/home/jiamingp/Diffusion_model/cosmo_diffusion_main"


def parse_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.replace("+", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def finite_summary(tensor) -> str:
    import torch

    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return "finite=0"
    return (
        f"finite={finite.numel()}/{tensor.numel()} "
        f"min={finite.min().item():.6g} max={finite.max().item():.6g} "
        f"mean={finite.float().mean().item():.6g}"
    )


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

    torch = install_torch_backend_compat(entry_point=__name__)
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

    model = None
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
        small_config["data"]["n_samples"] = [int(args.finite_samples_per_field)] * n_classes
        small_config["data"]["seed"] = [None] * n_classes
        data_out = utils.parse_config_data(small_config)
        dataset = data_out["data"] if isinstance(data_out, dict) else data_out
        sample = dataset[0]
        if not torch.isfinite(dataset.arrays).all():
            lines = ["Non-finite values found after transform/normalization:"]
            all_labels = dataset.labels.detach().cpu() if dataset.labels is not None else None
            for class_id in range(n_classes):
                if all_labels is None:
                    values = dataset.arrays
                else:
                    values = dataset.arrays[all_labels == class_id]
                lines.append(f"  class {class_id}: {finite_summary(values)}")
            raise RuntimeError("\n".join(lines))
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
            f"label_dtype={labels.dtype} class_ids={uniq} data_stats=({finite_summary(dataset.arrays)})",
            flush=True,
        )
        if not args.skip_training_step and model is not None:
            batch_size = min(4, len(dataset))
            batch_images = dataset.arrays[:batch_size].float()
            batch_labels = dataset.labels[:batch_size].long()
            scheduler_cls = getattr(diffusers, config["noise_scheduler"]["class"])
            scheduler = scheduler_cls(**copy.deepcopy(config["noise_scheduler"].get("kwargs", {})))
            noise = torch.randn_like(batch_images)
            timesteps = torch.randint(
                0,
                int(scheduler.config.num_train_timesteps),
                (batch_size,),
                dtype=torch.long,
            )
            noisy_images, target = optim.noise_and_target(scheduler, batch_images, noise, timesteps)
            with torch.no_grad():
                pred = model(noisy_images, timestep=timesteps, class_labels=batch_labels, return_dict=False)[0]
                per_sample_mse = torch.nn.functional.mse_loss(pred, target, reduction="none").mean(
                    dim=list(range(1, pred.ndim))
                )
                loss = per_sample_mse.mean()
            checks = {
                "batch_images": batch_images,
                "noisy_images": noisy_images,
                "target": target,
                "pred": pred,
                "loss": loss.reshape(1),
            }
            bad = [name for name, tensor in checks.items() if not torch.isfinite(tensor).all()]
            if bad:
                details = "\n".join(f"  {name}: {finite_summary(tensor)}" for name, tensor in checks.items())
                raise RuntimeError(f"Non-finite one-step training check: {bad}\n{details}")
            print(f"[preflight] one-step loss check ok: loss={loss.item():.6g}", flush=True)

    print("[preflight] nf_class_conditional_u128 runtime checks passed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--cosmodiff-dir", default=DEFAULT_COSMODIFF_DIR)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-model-forward", action="store_true")
    parser.add_argument("--skip-small-data-load", action="store_true")
    parser.add_argument("--skip-training-step", action="store_true")
    parser.add_argument("--finite-samples-per-field", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    preflight(parse_args())


if __name__ == "__main__":
    main()
