#!/usr/bin/env python
"""Prepare Nick-default generalizability configs with more, less-correlated data.

This sweep is intended to reproduce the generalizability plots with a stronger
data setup than the original single-file LH z=0.0 runs:

- combine LH and CV grids
- combine z=0.0, z=1.0, and z=2.0
- use zthin=8, so each 3D cube contributes 16 less-correlated 2D slices
- keep the merged Nick-default diffusion/training recipe
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SWEEP_NAME = "nf_generalize_nick_data"
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
DATA_ROOT = "/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG"
ZTHIN = 8
SLICES_PER_VOLUME = 128 // ZTHIN
EMA_SIGMA_RELS = [0.02, 0.10]


DATA_SOURCES = [
    {
        "tag": "lh_z0p0",
        "dataset": "LH",
        "redshift": 0.0,
        "path": f"{DATA_ROOT}/Grids_HI_IllustrisTNG_LH_128_z=0.0.npy",
        "capacity": 1000,
    },
    {
        "tag": "cv_z0p0",
        "dataset": "CV",
        "redshift": 0.0,
        "path": f"{DATA_ROOT}/Grids_HI_IllustrisTNG_CV_128_z=0.0.npy",
        "capacity": 27,
    },
    {
        "tag": "lh_z1p0",
        "dataset": "LH",
        "redshift": 1.0,
        "path": f"{DATA_ROOT}/Grids_HI_IllustrisTNG_LH_128_z=1.0.npy",
        "capacity": 1000,
    },
    {
        "tag": "cv_z1p0",
        "dataset": "CV",
        "redshift": 1.0,
        "path": f"{DATA_ROOT}/Grids_HI_IllustrisTNG_CV_128_z=1.0.npy",
        "capacity": 27,
    },
    {
        "tag": "lh_z2p0",
        "dataset": "LH",
        "redshift": 2.0,
        "path": f"{DATA_ROOT}/Grids_HI_IllustrisTNG_LH_128_z=2.0.npy",
        "capacity": 1000,
    },
    {
        "tag": "cv_z2p0",
        "dataset": "CV",
        "redshift": 2.0,
        "path": f"{DATA_ROOT}/Grids_HI_IllustrisTNG_CV_128_z=2.0.npy",
        "capacity": 27,
    },
]


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


ARCH = {
    "arch": "u128",
    "arch_label": "UNet-128",
    "block_out_channels": [32, 64, 128],
    "norm_num_groups": 32,
    "batch_size": 32,
}


def allocate_source_counts(target_2d: int) -> list[dict[str, Any]]:
    """Allocate training volumes across source files without exceeding caps."""
    target_volumes = target_2d // SLICES_PER_VOLUME
    if target_volumes * SLICES_PER_VOLUME != target_2d:
        raise ValueError(f"{target_2d=} must be divisible by {SLICES_PER_VOLUME=}.")

    total_capacity = sum(int(src["capacity"]) for src in DATA_SOURCES)
    if target_volumes > total_capacity:
        raise ValueError(f"{target_volumes=} exceeds available source capacity {total_capacity}.")

    counts = [0 for _ in DATA_SOURCES]
    remaining = target_volumes
    while remaining > 0:
        progressed = False
        for i, source in enumerate(DATA_SOURCES):
            if remaining == 0:
                break
            if counts[i] >= int(source["capacity"]):
                continue
            counts[i] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("Could not allocate requested source counts.")

    rows = []
    for source, count in zip(DATA_SOURCES, counts):
        if count <= 0:
            continue
        row = deepcopy(source)
        row["n_samples"] = count
        row["n_2d_slices"] = count * SLICES_PER_VOLUME
        rows.append(row)
    return rows


def run_name(dataset_tag: str) -> str:
    return f"nf_gen_nick_u128_{dataset_tag}"


def build_config(name: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
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
                "block_out_channels": ARCH["block_out_channels"],
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
                "norm_num_groups": ARCH["norm_num_groups"],
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
            "batch_size": ARCH["batch_size"],
            "shuffle": True,
            "checkpoint_every_n_epochs": 5,
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
            "n_samples": 512,
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
    for dataset_tag, target_2d in RUN_SIZES:
        name = run_name(dataset_tag)
        sources = allocate_source_counts(target_2d)
        n_train_volumes = sum(int(src["n_samples"]) for src in sources)
        actual_2d = sum(int(src["n_2d_slices"]) for src in sources)
        rows.append({
            "run_name": name,
            "arch": ARCH["arch"],
            "arch_label": ARCH["arch_label"],
            "variant_tag": "nick_default",
            "variant_label": "Nick default",
            "dataset_tag": dataset_tag,
            "dataset_group": "LH+CV z=0,1,2",
            "target_2d": target_2d,
            "actual_2d": actual_2d,
            "dataset_size": actual_2d,
            "n_train_simulations": n_train_volumes,
            "n_samples_simulations": n_train_volumes,
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
            "epochs": 100,
            "batch_size": ARCH["batch_size"],
            "beta_schedule": "squaredcos_cap_v2",
            "rescale_betas_zero_snr": True,
            "prediction_type": "v_prediction",
            "sigma_log_normal": None,
            "min_snr_gamma": 5.0,
            "ema_sigma_rels": EMA_SIGMA_RELS,
            "ema_burn_in": 1000,
            "config": f"local/{SWEEP_NAME}/configs/{name}.yaml",
            "checkpoint_dir": f"{CHECKPOINT_ROOT}/{name}_checkpoints",
            "sample_path": f"results/{SWEEP_NAME}/samples/{name}_seed{{seed}}_raw_train_full.npz",
            "note": "Nick-default u128 recipe with LH+CV and z=0/1/2, zthin=8.",
        })
    return rows


def expected_by_run() -> dict[str, dict[str, Any]]:
    return {
        row["run_name"]: build_config(row["run_name"], row["source_counts"])
        for row in iter_runs()
    }


def assert_config(path: Path, expected: dict[str, Any]) -> None:
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
        "augmentations.RandomRoll": config.get("augmentations", {}).get("RandomRoll", {}).get("dims") == [-1, -2],
        "model.block_out_channels": (
            config["model"]["kwargs"].get("block_out_channels")
            == expected["model"]["kwargs"]["block_out_channels"]
        ),
        "noise_scheduler.kwargs": config["noise_scheduler"]["kwargs"] == expected["noise_scheduler"]["kwargs"],
        "train.ema_sigma_rels": config["train"].get("ema_sigma_rels") == EMA_SIGMA_RELS,
        "train.ema_update_every": config["train"].get("ema_update_every") == 1,
        "train.min_snr_gamma": config["train"].get("min_snr_gamma") == 5.0,
        "train.sigma_log_normal": config["train"].get("sigma_log_normal") is None,
        "generate.exists": "generate" in config,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def selected_run_names(args: argparse.Namespace) -> set[str]:
    names = {row["run_name"] for row in iter_runs()}
    if args.run_name:
        names &= set(args.run_name)
    if args.dataset_tag:
        wanted = set(args.dataset_tag)
        names &= {row["run_name"] for row in iter_runs() if row["dataset_tag"] in wanted}
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--check-only", action="store_true", help="Validate existing configs without writing.")
    parser.add_argument("--print-runs", action="store_true", help="Print selected run names and exit.")
    parser.add_argument("--run-name", action="append", help="Optional run name. Repeatable.")
    parser.add_argument("--dataset-tag", action="append", help="Optional dataset tag such as d2p11. Repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_dir = project_dir / "local" / SWEEP_NAME / "configs"
    manifest_path = project_dir / "local" / SWEEP_NAME / "manifest.json"
    selected = selected_run_names(args)
    expected = expected_by_run()

    if args.print_runs:
        for row in iter_runs():
            if row["run_name"] in selected:
                print(row["run_name"])
        return

    if args.check_only:
        for name in selected:
            assert_config(config_dir / f"{name}.yaml", expected[name])
        print(f"Validated {len(selected)} {SWEEP_NAME} configs.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    for name, config in expected.items():
        if name not in selected:
            continue
        path = config_dir / f"{name}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"Wrote {path}")

    manifest = [row for row in iter_runs() if row["run_name"] in selected]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
