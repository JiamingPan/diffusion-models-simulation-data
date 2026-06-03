#!/usr/bin/env python
"""Runtime preflight for continuous HI bias-probe training configs."""

from __future__ import annotations

import argparse
import copy
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Bias-probe YAML config.")
    parser.add_argument("--cosmodiff-dir", required=True, help="Expected cosmo_diffusion checkout.")
    parser.add_argument("--skip-forward", action="store_true", help="Skip the tiny CPU model forward check.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available.")
    parser.add_argument(
        "--allow-cfg",
        action="store_true",
        help="Allow nonzero CFG settings for explicit CFG/guidance ablation configs.",
    )
    return parser.parse_args()


def require_file(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {label}: {p}")
    return p


def check_cosmodiff_data_loader(cfg: dict, utils: object, project_dir: Path) -> None:
    """Run the real cosmodiff data loader on a tiny memmap-backed config."""

    image_path = Path(cfg["data"]["img_path"])
    label_path = Path(cfg["data"]["label_path"])
    images = np.load(image_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    n = min(2, int(images.shape[0]), int(labels.shape[0]))
    if n <= 0:
        raise ValueError("Cannot build tiny data-loader preflight from empty arrays.")

    cache_dir = project_dir / "results" / "cache" / "nf_conditional_bias_probe_preflight"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = image_path.stem
    tiny_images_path = cache_dir / f"{tag}_tiny_images.npy"
    tiny_labels_path = cache_dir / f"{tag}_tiny_labels.npy"
    np.save(tiny_images_path, np.asarray(images[:n], dtype=np.float32))
    np.save(tiny_labels_path, np.asarray(labels[:n], dtype=np.float32))

    tiny_cfg = copy.deepcopy(cfg)
    tiny_cfg["global"] = dict(tiny_cfg.get("global", {}))
    tiny_cfg["global"]["device"] = "cpu"
    tiny_cfg["data"] = dict(tiny_cfg["data"])
    tiny_cfg["data"].update(
        {
            "img_path": str(tiny_images_path),
            "label_path": str(tiny_labels_path),
            "reshape": None,
            "zthin": 1,
            "n_samples": None,
            "seed": None,
            "keep_on_cpu": True,
        }
    )
    data_out = utils.parse_config_data(tiny_cfg)
    dataset = data_out["data"] if isinstance(data_out, dict) else data_out
    sample = dataset[0]
    sample_labels = sample.get("labels")
    if sample_labels is None:
        raise ValueError("cosmodiff data loader returned no labels for continuous config.")
    import torch

    if not torch.is_floating_point(sample_labels):
        raise ValueError(f"Continuous labels must be floating point after patch, got {sample_labels.dtype}.")
    if sample_labels.numel() != 6:
        raise ValueError(f"Expected 6 conditioning parameters, got shape {tuple(sample_labels.shape)}.")
    print(
        "[preflight] cosmodiff_parse_config_data_ok="
        f"len={len(dataset)} image_shape={tuple(sample['images'].shape)} label_dtype={sample_labels.dtype}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config_path = require_file(args.config, "config")
    cosmodiff_dir = Path(args.cosmodiff_dir).resolve()
    train_script = require_file(cosmodiff_dir / "scripts" / "cosmodiff_train.py", "cosmodiff_train.py")

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    print(f"[preflight] python={sys.executable}", flush=True)
    print(f"[preflight] config={config_path}", flush=True)
    print(f"[preflight] cosmodiff_train={train_script}", flush=True)
    print(f"[preflight] PYTHONPATH={os.environ.get('PYTHONPATH', '')}", flush=True)

    import torch

    print(f"[preflight] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[preflight] torch_file={Path(inspect.getsourcefile(torch)).resolve()}", flush=True)
    print(f"[preflight] torch_cuda_build={getattr(torch.version, 'cuda', None)}", flush=True)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this training job but torch.cuda.is_available() is false. "
            "Use the CUDA-enabled cosmodiff_nf_class environment on a GPU node."
        )
    print(f"[preflight] torch.xpu.exists={hasattr(torch, 'xpu')}", flush=True)
    if hasattr(torch, "xpu"):
        print(f"[preflight] torch.xpu.is_available={torch.xpu.is_available()}", flush=True)
    print(f"[preflight] torch.mps.exists={hasattr(torch, 'mps')}", flush=True)

    import diffusers
    from diffusers import UNet2DConditionModel

    print(f"[preflight] diffusers={getattr(diffusers, '__version__', 'unknown')}", flush=True)
    print(f"[preflight] diffusers_file={Path(inspect.getsourcefile(diffusers)).resolve()}", flush=True)

    from cosmodiff import optim, utils

    optim_path = Path(inspect.getsourcefile(optim)).resolve()
    utils_path = Path(inspect.getsourcefile(utils)).resolve()
    print(f"[preflight] cosmodiff.optim={optim_path}", flush=True)
    print(f"[preflight] cosmodiff.utils={utils_path}", flush=True)
    if cosmodiff_dir not in optim_path.parents or cosmodiff_dir not in utils_path.parents:
        raise RuntimeError(f"Imported cosmodiff outside expected checkout: {cosmodiff_dir}")

    if cfg["model"]["class"] != "UNet2DConditionModel":
        raise ValueError(f"Expected UNet2DConditionModel, got {cfg['model']['class']}")
    if cfg["train"].get("conditioning") != "continuous":
        raise ValueError(f"Expected train.conditioning=continuous, got {cfg['train'].get('conditioning')}")
    if not args.allow_cfg:
        if cfg["train"].get("cfg_dropout") != 0.0:
            raise ValueError(f"Expected CFG off for v1, got cfg_dropout={cfg['train'].get('cfg_dropout')}")
        if cfg["generate"].get("guidance_scale") is not None:
            raise ValueError(f"Expected guidance_scale=None for v1, got {cfg['generate'].get('guidance_scale')}")

    image_path = require_file(cfg["data"]["img_path"], "training image array")
    label_path = require_file(cfg["data"]["label_path"], "training label array")
    labels = np.load(label_path, mmap_mode="r")
    images = np.load(image_path, mmap_mode="r")
    print(f"[preflight] image_shape={images.shape} image_dtype={images.dtype}", flush=True)
    print(f"[preflight] label_shape={labels.shape} label_dtype={labels.dtype}", flush=True)
    if labels.ndim != 2 or labels.shape[1] != 6:
        raise ValueError(f"Expected labels shape (N, 6), got {labels.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}")

    output_dir = Path(cfg["io"]["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[preflight] output_dir_parent_ok={output_dir.parent}", flush=True)

    project_dir = Path(os.environ.get("PROJECT_DIR", ".")).resolve()
    check_cosmodiff_data_loader(cfg, utils, project_dir)

    if not args.skip_forward:
        kwargs = dict(cfg["model"]["kwargs"])
        model = UNet2DConditionModel(**kwargs).eval()
        x = torch.zeros(1, int(kwargs["in_channels"]), int(kwargs["sample_size"]), int(kwargs["sample_size"]))
        t = torch.zeros(1, dtype=torch.long)
        cond_dim = int(kwargs["encoder_hid_dim"])
        encoder_hidden_states = torch.zeros(1, 1, cond_dim)
        with torch.no_grad():
            out = model(x, t, encoder_hidden_states=encoder_hidden_states).sample
        print(f"[preflight] tiny_forward_output_shape={tuple(out.shape)}", flush=True)

    print("[preflight] nf_conditional_bias_probe runtime checks passed", flush=True)


if __name__ == "__main__":
    main()
