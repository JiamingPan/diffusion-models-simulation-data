#!/usr/bin/env python
"""Prepare Fig. 2 style CAMELS generalization configs.

This sweep is meant to reproduce the training-size experiment in
Zhang et al. (arXiv:2310.05264) in our CAMELS/HI setting:

- train separate models for powers-of-two dataset sizes
- run both u64 and u128 model widths
- use a fixed optimizer-update budget across N
- remove data augmentation so low-N memorization is not hidden by shifts/flips

The old fixed-epoch sweep is still useful for quality checks, but it gives
tiny-N runs far fewer optimizer updates. This sweep fixes that.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from prepare_nf_generalize_nick_data_configs import (
    DATA_SOURCES,
    EMA_SIGMA_RELS,
    SLICES_PER_VOLUME,
    ZTHIN,
    allocate_source_counts,
)


SWEEP_NAME = "nf_generalize_fig2"
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
TARGET_UPDATES = 200_000
CHECKPOINT_EVERY_UPDATES = 20_000
GENERATE_N_SAMPLES = 512


ARCHES = {
    "u64": {
        "label": "UNet-64",
        "block_out_channels": [16, 32, 64],
        "norm_num_groups": 16,
        "batch_size": 32,
    },
    "u128": {
        "label": "UNet-128",
        "block_out_channels": [32, 64, 128],
        "norm_num_groups": 32,
        "batch_size": 32,
    },
}


RUN_SIZES = [
    ("d2p06", 64),
    ("d2p07", 128),
    ("d2p08", 256),
    ("d2p09", 512),
    ("d2p10", 1024),
    ("d2p11", 2048),
    ("d2p12", 4096),
    ("d2p13", 8192),
    ("d2p14", 16384),
    ("d2p15", 32768),
]


def run_name(arch: str, dataset_tag: str) -> str:
    return f"nf_fig2_{arch}_{dataset_tag}_noaug_200k"


def steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(int(dataset_size) / int(batch_size)))


def epochs_for(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(TARGET_UPDATES / steps_per_epoch(dataset_size, batch_size)))


def checkpoint_epochs_for(dataset_size: int, batch_size: int) -> int:
    return max(1, round(CHECKPOINT_EVERY_UPDATES / steps_per_epoch(dataset_size, batch_size)))


def build_config(name: str, arch: str, dataset_size: int, sources: list[dict[str, Any]]) -> dict[str, Any]:
    arch_cfg = ARCHES[arch]
    batch_size = int(arch_cfg["batch_size"])
    num_epochs = epochs_for(dataset_size, batch_size)
    checkpoint_every_n_epochs = checkpoint_epochs_for(dataset_size, batch_size)
    return {
        "global": {
            "device": "cuda",
            "dtype": "float32",
        },
        "io": {
            "output_dir": f"{CHECKPOINT_ROOT}/{name}_checkpoints",
        },
        "data": {
            "img_path": [src["path"] for src in sources],
            "img_read_fn": "npy_read_fn",
            "label_path": None,
            "label_read_fn": "npy_read_fn",
            "reshape": "2d",
            "zthin": ZTHIN,
            "n_samples": [int(src["n_samples"]) for src in sources],
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
                "norm_num_groups": arch_cfg["norm_num_groups"],
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
            "num_epochs": int(num_epochs),
            "batch_size": batch_size,
            "shuffle": True,
            "checkpoint_every_n_epochs": int(checkpoint_every_n_epochs),
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
            "n_samples": GENERATE_N_SAMPLES,
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
    rows: list[dict[str, Any]] = []
    for arch, arch_cfg in ARCHES.items():
        for dataset_tag, target_2d in RUN_SIZES:
            name = run_name(arch, dataset_tag)
            sources = allocate_source_counts(target_2d)
            batch_size = int(arch_cfg["batch_size"])
            spe = steps_per_epoch(target_2d, batch_size)
            epochs = epochs_for(target_2d, batch_size)
            actual_updates = spe * epochs
            rows.append(
                {
                    "run_name": name,
                    "arch": arch,
                    "arch_label": arch_cfg["label"],
                    "variant_tag": "noaug_fixed_updates",
                    "variant_label": "No augmentation, fixed 200k updates",
                    "dataset_tag": dataset_tag,
                    "dataset_group": "LH+CV z=0,1,2",
                    "target_2d": target_2d,
                    "actual_2d": target_2d,
                    "dataset_size": target_2d,
                    "n_train_simulations": sum(int(src["n_samples"]) for src in sources),
                    "n_samples_simulations": sum(int(src["n_samples"]) for src in sources),
                    "zthin": ZTHIN,
                    "slices_per_sim": SLICES_PER_VOLUME,
                    "source_counts": [
                        {
                            "tag": src["tag"],
                            "path": src["path"],
                            "n_samples": int(src["n_samples"]),
                            "n_2d_slices": int(src["n_2d_slices"]),
                        }
                        for src in sources
                    ],
                    "epochs": int(epochs),
                    "steps_per_epoch": int(spe),
                    "target_updates": TARGET_UPDATES,
                    "actual_updates": int(actual_updates),
                    "checkpoint_every_updates": CHECKPOINT_EVERY_UPDATES,
                    "checkpoint_every_n_epochs": int(checkpoint_epochs_for(target_2d, batch_size)),
                    "batch_size": batch_size,
                    "beta_schedule": "squaredcos_cap_v2",
                    "rescale_betas_zero_snr": True,
                    "prediction_type": "v_prediction",
                    "sigma_log_normal": None,
                    "min_snr_gamma": 5.0,
                    "ema_sigma_rels": EMA_SIGMA_RELS,
                    "ema_burn_in": 1000,
                    "augmentations": None,
                    "config": f"local/{SWEEP_NAME}/configs/{name}.yaml",
                    "checkpoint_dir": f"{CHECKPOINT_ROOT}/{name}_checkpoints",
                    "sample_path": f"results/{SWEEP_NAME}/samples/{name}_seed{{seed}}_raw_train_full.npz",
                    "note": "Fig.2-style run: no augmentation and about 200k optimizer updates.",
                }
            )
    return rows


def expected_by_run() -> dict[str, dict[str, Any]]:
    out = {}
    for row in iter_runs():
        out[row["run_name"]] = build_config(
            row["run_name"],
            row["arch"],
            int(row["dataset_size"]),
            deepcopy(row["source_counts"]),
        )
    return out


def assert_config(path: Path, expected: dict[str, Any], row: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)

    checks = {
        "data.img_path_list": isinstance(config["data"].get("img_path"), list),
        "data.n_samples_list": isinstance(config["data"].get("n_samples"), list),
        "data.zthin": config["data"].get("zthin") == ZTHIN,
        "data.transform": config["data"].get("transform") == ["log"],
        "data.normalization": config["data"].get("normalization") == "tanh",
        "data.target_size": sum(config["data"]["n_samples"]) * SLICES_PER_VOLUME == row["dataset_size"],
        "augmentations.absent": "augmentations" not in config,
        "model.block_out_channels": (
            config["model"]["kwargs"].get("block_out_channels")
            == expected["model"]["kwargs"]["block_out_channels"]
        ),
        "noise_scheduler.kwargs": config["noise_scheduler"]["kwargs"] == expected["noise_scheduler"]["kwargs"],
        "train.num_epochs": config["train"].get("num_epochs") == row["epochs"],
        "train.checkpoint_every_n_epochs": (
            config["train"].get("checkpoint_every_n_epochs") == row["checkpoint_every_n_epochs"]
        ),
        "train.ema_sigma_rels": config["train"].get("ema_sigma_rels") == EMA_SIGMA_RELS,
        "train.min_snr_gamma": config["train"].get("min_snr_gamma") == 5.0,
        "train.sigma_log_normal": config["train"].get("sigma_log_normal") is None,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def selected_run_names(args: argparse.Namespace) -> set[str]:
    rows = iter_runs()
    names = {row["run_name"] for row in rows}
    if args.run_name:
        names &= set(args.run_name)
    if args.arch:
        wanted = set(args.arch)
        names &= {row["run_name"] for row in rows if row["arch"] in wanted}
    if args.dataset_tag:
        wanted = set(args.dataset_tag)
        names &= {row["run_name"] for row in rows if row["dataset_tag"] in wanted}
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--check-only", action="store_true", help="Validate existing configs without writing.")
    parser.add_argument("--print-runs", action="store_true", help="Print selected run names and exit.")
    parser.add_argument("--run-name", action="append", help="Optional run name. Repeatable.")
    parser.add_argument("--arch", choices=sorted(ARCHES), action="append", help="Restrict to architecture. Repeatable.")
    parser.add_argument("--dataset-tag", action="append", help="Restrict to dataset tag such as d2p11. Repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_dir = project_dir / "local" / SWEEP_NAME / "configs"
    manifest_path = project_dir / "local" / SWEEP_NAME / "manifest.json"
    selected = selected_run_names(args)
    expected = expected_by_run()
    rows = [row for row in iter_runs() if row["run_name"] in selected]

    if args.print_runs:
        for row in rows:
            print(row["run_name"])
        return

    if args.check_only:
        for row in rows:
            assert_config(config_dir / f"{row['run_name']}.yaml", expected[row["run_name"]], row)
        print(f"Validated {len(rows)} {SWEEP_NAME} configs.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = config_dir / f"{row['run_name']}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(expected[row["run_name"]], f, sort_keys=False)
        print(f"Wrote {path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
