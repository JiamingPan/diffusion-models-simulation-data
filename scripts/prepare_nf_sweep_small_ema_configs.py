#!/usr/bin/env python
"""Prepare a focused Nick-default run for very small post-hoc EMA anchors.

This intentionally avoids the broader ``sigma_log_normal`` sweep.  The goal is
to test whether EMA only helps at much shorter averaging windows for the CAMELS
field metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


IMG_PATH = "/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/Grids_HI_IllustrisTNG_LH_128_z=0.0.npy"
CHECKPOINT_ROOT = "/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_sweep_small_ema"
EMA_SIGMA_RELS = [0.005, 0.05]
SWEEP_NAME = "nf_sweep_small_ema"


RUNS = [
    {
        "run_name": "nf_small_ema_u128_n500_e100_nick_default",
        "arch": "u128",
        "arch_label": "U128",
        "variant_tag": "nick_default",
        "variant_label": "Nick default + small EMA anchors",
        "block_out_channels": [32, 64, 128],
        "norm_num_groups": 32,
        "batch_size": 32,
    },
]


def build_config(row: dict[str, Any]) -> dict[str, Any]:
    name = row["run_name"]
    return {
        "global": {
            "device": "cuda",
            "dtype": "float32",
        },
        "io": {
            "output_dir": f"{CHECKPOINT_ROOT}/{name}_checkpoints",
        },
        "data": {
            "img_path": IMG_PATH,
            "img_read_fn": "npy_read_fn",
            "label_path": None,
            "label_read_fn": "npy_read_fn",
            "reshape": "2d",
            "zthin": 4,
            "n_samples": 500,
            "seed": None,
            "keep_on_cpu": True,
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
            "transform": ["log"],
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
                "block_out_channels": row["block_out_channels"],
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
                "norm_num_groups": row["norm_num_groups"],
            },
        },
        "noise_scheduler": {
            "class": "DDPMScheduler",
            "kwargs": {
                "num_train_timesteps": 500,
                "beta_schedule": "squaredcos_cap_v2",
                "rescale_betas_zero_snr": True,
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
            "batch_size": row["batch_size"],
            "shuffle": True,
            # Finer than Nick's default so very small post-hoc EMA targets are
            # not reconstructed from overly sparse snapshots.
            "checkpoint_every_n_epochs": 2,
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 1,
            "dataloader_num_workers": 0,
            "max_grad_norm": 1.0,
            "conditioning": "discrete",
            "cfg_dropout": 0.0,
            "ema_sigma_rels": EMA_SIGMA_RELS,
            "ema_update_every": 1,
            "ema_burn_in": 1000,
            "min_snr_gamma": 5.0,
            "sigma_log_normal": None,
            "verbose": True,
            "force_cpu": False,
            "pin_memory": False,
        },
        "generate": {
            "scheduler": None,
            "num_steps": None,
            "s_churn": None,
            "s_tmin": None,
            "s_tmax": None,
            "s_noise": None,
            "n_samples": 64,
            "batch_size": None,
            "image_shape": None,
            "conditioning": "discrete",
            "labels": None,
            "continuous_labels": None,
            "guidance_scale": None,
            "ema_sigma_rel": None,
            "seed": None,
            "device": None,
        },
    }


def iter_runs() -> list[dict[str, Any]]:
    rows = []
    for row in RUNS:
        rows.append(
            {
                "run_name": row["run_name"],
                "arch": row["arch"],
                "arch_label": row["arch_label"],
                "variant_tag": row["variant_tag"],
                "variant_label": row["variant_label"],
                "n_train_simulations": 500,
                "zthin": 4,
                "expected_2d_slices": 16000,
                "epochs": 100,
                "batch_size": row["batch_size"],
                "beta_schedule": "squaredcos_cap_v2",
                "rescale_betas_zero_snr": True,
                "prediction_type": "v_prediction",
                "sigma_log_normal": None,
                "min_snr_gamma": 5.0,
                "ema_sigma_rels": EMA_SIGMA_RELS,
                "ema_burn_in": 1000,
                "checkpoint_every_n_epochs": 2,
                "config": f"local/{SWEEP_NAME}/configs/{row['run_name']}.yaml",
                "checkpoint_dir": f"{CHECKPOINT_ROOT}/{row['run_name']}_checkpoints",
                "note": "Nick-default recipe with very small native post-hoc EMA anchors.",
            }
        )
    return rows


def expected_by_run() -> dict[str, dict[str, Any]]:
    return {row["run_name"]: build_config(row) for row in RUNS}


def assert_config(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)

    checks = {
        "data.transform": config["data"].get("transform") == ["log"],
        "data.normalization": config["data"].get("normalization") == "tanh",
        "augmentations.RandomRoll": config.get("augmentations", {}).get("RandomRoll", {}).get("dims") == [-1, -2],
        "augmentations.no_RandomFlip": "RandomFlip" not in config.get("augmentations", {}),
        "noise_scheduler.kwargs": config["noise_scheduler"]["kwargs"] == expected["noise_scheduler"]["kwargs"],
        "train.ema_sigma_rels": config["train"].get("ema_sigma_rels") == EMA_SIGMA_RELS,
        "train.checkpoint_every_n_epochs": config["train"].get("checkpoint_every_n_epochs") == 2,
        "train.min_snr_gamma": config["train"].get("min_snr_gamma") == 5.0,
        "train.sigma_log_normal": config["train"].get("sigma_log_normal") is None,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--check-only", action="store_true", help="Validate existing configs without writing.")
    parser.add_argument("--print-runs", action="store_true", help="Print selected run names and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_dir = project_dir / "local" / SWEEP_NAME / "configs"
    manifest_path = project_dir / "local" / SWEEP_NAME / "manifest.json"
    expected = expected_by_run()

    if args.print_runs:
        for row in iter_runs():
            print(row["run_name"])
        return

    if args.check_only:
        for name in expected:
            assert_config(config_dir / f"{name}.yaml", expected[name])
        print(f"Validated {len(expected)} {SWEEP_NAME} configs.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    for name, config in expected.items():
        path = config_dir / f"{name}.yaml"
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
