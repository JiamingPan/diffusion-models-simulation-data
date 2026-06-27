#!/usr/bin/env python
"""Runtime preflight for the Fig.2 DiT sweep.

Run this inside the same Great Lakes environment as the DiT training Slurm job
before submitting the full array.  It intentionally checks the exact fragile
path that failed before: ``data.constant_label`` must create integer labels,
and ``DiTTransformer2DModel`` must receive those labels as ``class_labels``.

The check loads only a few real CAMELS slices, instantiates the real DiT-base
model, and runs one no-grad noisy forward/loss calculation.  It does not write
checkpoints or start training.
"""

from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path

import torch
import yaml


DEFAULT_COSMODIFF_DIR = "/home/jiamingp/Diffusion_model/cosmo_diffusion_main"


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def finite_summary(tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    return (
        f"shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.4g} "
        f"max={tensor.max().item():.4g} "
        f"mean={tensor.float().mean().item():.4g}"
    )


def source_has(path: Path, *needles: str) -> bool:
    text = path.read_text()
    return any(needle in text for needle in needles)


def precheck(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    cosmodiff_dir = Path(args.cosmodiff_dir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    config_path = config_path.resolve()

    with config_path.open() as f:
        config = yaml.safe_load(f)

    if config["model"].get("class") != "DiTTransformer2DModel":
        raise RuntimeError(f"Expected DiTTransformer2DModel, got {config['model'].get('class')!r}")
    if config["data"].get("constant_label") != 0:
        raise RuntimeError(f"Expected data.constant_label=0, got {config['data'].get('constant_label')!r}")
    if config["train"].get("conditioning") != "discrete":
        raise RuntimeError(f"Expected train.conditioning=discrete, got {config['train'].get('conditioning')!r}")

    data_paths = [Path(p) for p in as_list(config["data"].get("img_path"))]
    missing = [str(p) for p in data_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing data files:\n" + "\n".join(missing))

    import diffusers
    from cosmodiff import optim, utils

    optim_path = Path(inspect.getsourcefile(optim)).resolve()
    utils_path = Path(inspect.getsourcefile(utils)).resolve()
    if cosmodiff_dir not in optim_path.parents:
        raise RuntimeError(f"Imported cosmodiff.optim from {optim_path}, expected under {cosmodiff_dir}")
    if cosmodiff_dir not in utils_path.parents:
        raise RuntimeError(f"Imported cosmodiff.utils from {utils_path}, expected under {cosmodiff_dir}")

    if not source_has(utils_path, "codex constant-label patch", "constant_label = data_cfg.get"):
        raise RuntimeError("cosmodiff.utils is missing constant-label support for DiT labels.")
    if not source_has(optim_path, "codex DiT class-label patch", "class_labels=batch_labels", "class_labels=labels"):
        raise RuntimeError("cosmodiff.optim.train is missing the DiT class-label forward path.")

    print(f"[dit-precheck] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[dit-precheck] diffusers={getattr(diffusers, '__version__', 'unknown')}", flush=True)
    print(f"[dit-precheck] cosmodiff.optim={optim_path}", flush=True)
    print(f"[dit-precheck] cosmodiff.utils={utils_path}", flush=True)

    small_config = copy.deepcopy(config)
    small_config.setdefault("global", {})["device"] = "cpu"
    small_config["data"]["keep_on_cpu"] = True
    original_n_samples = as_list(small_config["data"].get("n_samples"))
    if not original_n_samples:
        raise RuntimeError("Config must use list-valued data.n_samples for this sweep.")
    small_config["data"]["n_samples"] = [
        max(0, min(int(n), int(args.source_samples_per_file))) for n in original_n_samples
    ]
    if not any(small_config["data"]["n_samples"]):
        raise RuntimeError(f"Small-load n_samples became all zero: {small_config['data']['n_samples']}")

    dataset = utils.parse_config_data(small_config)
    sample = dataset[0]
    if "labels" not in sample:
        raise RuntimeError("Dataset sample has no labels; DiT would crash with class_labels=None.")
    if dataset.labels is None:
        raise RuntimeError("Dataset.labels is None; data.constant_label did not create label tensor.")
    labels_cpu = dataset.labels.detach().cpu()
    unique_labels = sorted(int(x) for x in torch.unique(labels_cpu).tolist())
    if unique_labels != [0]:
        raise RuntimeError(f"Expected one null class label [0], got {unique_labels}")
    if not torch.is_floating_point(dataset.arrays):
        raise RuntimeError(f"Expected floating image tensor, got {dataset.arrays.dtype}")
    if not torch.isfinite(dataset.arrays).all():
        raise RuntimeError(f"Non-finite image tensor after load: {finite_summary(dataset.arrays)}")
    print(
        "[dit-precheck] data load ok: "
        f"len={len(dataset)} image_shape={tuple(sample['images'].shape)} "
        f"label_dtype={sample['labels'].dtype} labels={unique_labels} "
        f"stats=({finite_summary(dataset.arrays)})",
        flush=True,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")

    model_cls = getattr(diffusers, config["model"]["class"])
    model = model_cls(**copy.deepcopy(config["model"].get("kwargs", {}))).to(device).eval()
    scheduler_cls = getattr(diffusers, config["noise_scheduler"]["class"])
    scheduler = scheduler_cls(**copy.deepcopy(config["noise_scheduler"].get("kwargs", {})))

    batch_size = min(int(args.batch_size), len(dataset))
    images = dataset.arrays[:batch_size].float().to(device)
    labels = labels_cpu[:batch_size].long().to(device)
    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps),
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn_like(images)
    noisy_images = scheduler.add_noise(images, noise, timesteps)

    with torch.no_grad():
        pred = model(
            noisy_images,
            timestep=timesteps,
            class_labels=labels,
            return_dict=False,
        )[0]
        loss = torch.nn.functional.mse_loss(pred, noise)

    if tuple(pred.shape) != tuple(images.shape):
        raise RuntimeError(f"DiT output shape mismatch: pred={tuple(pred.shape)} images={tuple(images.shape)}")
    for name, tensor in {
        "images": images,
        "noisy_images": noisy_images,
        "pred": pred,
        "loss": loss.reshape(1),
    }.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite {name}: {finite_summary(tensor)}")

    print(
        "[dit-precheck] DiT labeled forward ok: "
        f"batch={batch_size} output_shape={tuple(pred.shape)} loss={loss.item():.6g}",
        flush=True,
    )
    print("[dit-precheck] PASS: safe to submit the full DiT train array.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--cosmodiff-dir", default=DEFAULT_COSMODIFF_DIR)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-samples-per-file", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> None:
    precheck(parse_args())


if __name__ == "__main__":
    main()
