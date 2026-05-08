#!/usr/bin/env python
"""Prepare U64/U128 Figure-1 LH configs in the new normalization format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


IMG_PATH = "/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/Grids_HI_IllustrisTNG_LH_128_z=0.0.npy"
CHECKPOINT_ROOT = "/scratch/huterer_root/huterer0/jiamingp/saved_runs/fig1_lh"

RUN_SIZES = [
    ("d2p06", 64, 64, 2, None),
    ("d2p07", 128, 128, 4, None),
    ("d2p08", 256, 256, 8, None),
    ("d2p09", 512, 512, 16, None),
    ("d2p10", 1024, 1024, 32, None),
    ("d2p11", 2048, 2048, 64, None),
    ("d2p12", 4096, 4096, 128, None),
    ("d2p13", 8192, 8192, 256, None),
    ("d2p14", 16384, 16384, 512, None),
    ("d2p15_full", 32768, 32000, None, "full LH set: 1000 simulations -> 32000 slices"),
]

ARCHES = {
    "u64": {
        "label": "UNet-64",
        "block_out_channels": [16, 32, 64],
        "batch_size": 32,
    },
    "u128": {
        "label": "UNet-128",
        "block_out_channels": [32, 64, 128],
        "batch_size": 32,
    },
}


def build_config(run_name: str, arch: str, n_samples: int | None) -> dict[str, Any]:
    arch_cfg = ARCHES[arch]
    return {
        "global": {
            "device": "cuda",
            "dtype": "float32",
        },
        "io": {
            "output_dir": f"{CHECKPOINT_ROOT}/{run_name}_checkpoints",
        },
        "data": {
            "img_path": IMG_PATH,
            "img_read_fn": "npy_read_fn",
            "label_path": None,
            "label_read_fn": "npy_read_fn",
            "two_dim": True,
            "zthin": 4,
            "n_samples": n_samples,
            "seed": None,
            "keep_on_cpu": True,
            "log": True,
            "normalization": "tanh",
            "norm_kwargs": {
                "center": None,
                "xmax": None,
                "alpha": 0.8,
                "beta": 10.0,
                "delta": 1.0,
                "gamma": 1.0,
                "sigma": 1.5,
            },
        },
        "augmentations": {
            "RandomRoll": {
                "size": 128,
                "dims": [-1, -2],
            },
        },
        "model": {
            "class": "UNet2DModel",
            "kwargs": {
                "sample_size": 128,
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 2,
                "block_out_channels": arch_cfg["block_out_channels"],
                "down_block_types": [
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                ],
                "up_block_types": [
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ],
                "norm_num_groups": 32,
            },
        },
        "noise_scheduler": {
            "class": "DDPMScheduler",
            "kwargs": {
                "num_train_timesteps": 500,
                "beta_schedule": "sigmoid",
                "prediction_type": "v_prediction",
                "clip_sample": False,
                "thresholding": False,
                "sample_max_value": 2.0,
            },
        },
        "optimizer": {
            "class": "AdamW",
            "kwargs": {
                "lr": 1.0e-4,
                "weight_decay": 1.0e-2,
            },
        },
        "lr_scheduler": {
            "class": "CosineAnnealingWarmRestarts",
            "kwargs": {
                "T_0": 4000,
                "eta_min": 1.0e-7,
            },
        },
        "train": {
            "num_epochs": 100,
            "batch_size": arch_cfg["batch_size"],
            "shuffle": True,
            "checkpoint_every_n_epochs": 5,
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 1,
            "dataloader_num_workers": 1,
            "max_grad_norm": 1.0,
            "verbose": True,
            "force_cpu": False,
            "pin_memory": False,
            "conditioning": "discrete",
            "cfg_dropout": 0.0,
            "ema_sigma_rels": None,
            "ema_update_every": 100,
        },
    }


def iter_runs() -> list[dict[str, Any]]:
    rows = []
    for arch, arch_cfg in ARCHES.items():
        for tag, target_2d, actual_2d, n_samples, note in RUN_SIZES:
            run_name = f"fig1_lh_{arch}_{tag}"
            rows.append({
                "run_name": run_name,
                "arch": arch,
                "arch_label": arch_cfg["label"],
                "dataset_tag": tag,
                "target_2d": target_2d,
                "actual_2d": actual_2d,
                "n_samples_simulations": n_samples,
                "zthin": 4,
                "slices_per_sim": 32,
                "epochs": 100,
                "batch_size": arch_cfg["batch_size"],
                "gradient_accumulation_steps": 1,
                "config": f"local/fig1_lh/configs/{run_name}.yaml",
                "checkpoint_dir": f"{CHECKPOINT_ROOT}/{run_name}_checkpoints",
                "note": note,
            })
    return rows


def expected_by_run() -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for row in iter_runs():
        expected[row["run_name"]] = build_config(
            row["run_name"],
            row["arch"],
            row["n_samples_simulations"],
        )
    return expected


def assert_new_format(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)

    checks = {
        "data.normalization": config["data"].get("normalization") == "tanh",
        "data.keep_on_cpu": config["data"].get("keep_on_cpu") is True,
        "augmentations.RandomRoll.size": config.get("augmentations", {}).get("RandomRoll", {}).get("size") == 128,
        "augmentations.no_RandomFlip": "RandomFlip" not in config.get("augmentations", {}),
        "model.block_out_channels": (
            config["model"]["kwargs"].get("block_out_channels")
            == expected["model"]["kwargs"]["block_out_channels"]
        ),
        "noise_scheduler.num_train_timesteps": (
            config["noise_scheduler"]["kwargs"].get("num_train_timesteps") == 500
        ),
        "noise_scheduler.beta_schedule": (
            config["noise_scheduler"]["kwargs"].get("beta_schedule") == "sigmoid"
        ),
        "noise_scheduler.prediction_type": (
            config["noise_scheduler"]["kwargs"].get("prediction_type") == "v_prediction"
        ),
        "lr_scheduler.class": config["lr_scheduler"].get("class") == "CosineAnnealingWarmRestarts",
        "train.conditioning": config["train"].get("conditioning") == "discrete",
        "train.ema_update_every": config["train"].get("ema_update_every") == 100,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            f"{path} is not in the new reproducibility config format. "
            f"Failed checks: {', '.join(failed)}. "
            "Run: python scripts/prepare_repro_u64_u128_configs.py"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--check-only", action="store_true", help="Validate existing configs without writing.")
    parser.add_argument("--run-name", action="append", help="Optional run name to validate/write. Repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_dir = project_dir / "local" / "fig1_lh" / "configs"
    manifest_path = project_dir / "local" / "fig1_lh" / "manifest.json"
    expected = expected_by_run()
    selected = set(args.run_name or expected)

    if args.check_only:
        for run_name in selected:
            assert_new_format(config_dir / f"{run_name}.yaml", expected[run_name])
        print(f"Validated {len(selected)} configs in new format.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    for run_name, config in expected.items():
        if run_name not in selected:
            continue
        path = config_dir / f"{run_name}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"Wrote {path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(iter_runs(), f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
