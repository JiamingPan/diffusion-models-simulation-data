#!/usr/bin/env python
"""Prepare a focused u128 augmentation sweep.

This sweep keeps Nicholas' default u128 recipe fixed and changes only the data
augmentation pipeline.  It is intentionally separate from
``nf_sweep_ema_sigma`` so older checkpoint folders are not reused with new
augmentation semantics.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


IMG_PATH = "/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/Grids_HI_IllustrisTNG_LH_128_z=0.0.npy"
CHECKPOINT_ROOT = "/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_sweep_aug"
EMA_SIGMA_RELS = [0.16, 0.25]

ARCHES = {
    "u128": {
        "label": "U128",
        "block_out_channels": [32, 64, 128],
        "norm_num_groups": 32,
        "batch_size": 32,
    },
}

BASE_AUGMENTATIONS = {
    "RandomRoll": {
        "size": 128,
        "dims": [-1, -2],
    },
    "RandomFlip": {
        "dims": [-1, -2],
    },
}

VARIANTS = [
    {
        "tag": "roll_flip",
        "label": "Roll + flip",
        "augmentations": BASE_AUGMENTATIONS,
        "note": "Clean control for the current roll+flip augmentation with fixed RandomFlip semantics.",
    },
    {
        "tag": "roll_flip_rot90",
        "label": "Roll + flip + rot90",
        "augmentations": {
            **BASE_AUGMENTATIONS,
            "RandomRot90": {
                "dims": [-1, -2],
            },
        },
        "note": "Adds random 0/90/180/270 degree rotations.",
    },
    {
        "tag": "roll_dihedral",
        "label": "Roll + dihedral",
        "augmentations": {
            "RandomRoll": {
                "size": 128,
                "dims": [-1, -2],
            },
            "RandomDihedral2D": {
                "dims": [-1, -2],
            },
        },
        "note": "Uses one random square symmetry instead of independent flip+rot90 draws.",
    },
]


def run_name(arch: str, tag: str) -> str:
    return f"nf_aug_{arch}_n500_e100_{tag}"


def build_config(name: str, arch: str, variant: dict[str, Any]) -> dict[str, Any]:
    arch_cfg = ARCHES[arch]
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
        "augmentations": deepcopy(variant["augmentations"]),
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
            "num_epochs": 100,
            "batch_size": arch_cfg["batch_size"],
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
    for arch, arch_cfg in ARCHES.items():
        for variant in VARIANTS:
            name = run_name(arch, variant["tag"])
            rows.append(
                {
                    "run_name": name,
                    "arch": arch,
                    "arch_label": arch_cfg["label"],
                    "variant_tag": variant["tag"],
                    "variant_label": variant["label"],
                    "augmentations": variant["augmentations"],
                    "n_train_simulations": 500,
                    "zthin": 4,
                    "expected_2d_slices": 16000,
                    "epochs": 100,
                    "batch_size": arch_cfg["batch_size"],
                    "beta_schedule": "squaredcos_cap_v2",
                    "rescale_betas_zero_snr": True,
                    "prediction_type": "v_prediction",
                    "sigma_log_normal": None,
                    "min_snr_gamma": 5.0,
                    "ema_sigma_rels": EMA_SIGMA_RELS,
                    "ema_burn_in": 1000,
                    "config": f"local/nf_sweep_aug/configs/{name}.yaml",
                    "checkpoint_dir": f"{CHECKPOINT_ROOT}/{name}_checkpoints",
                    "note": variant["note"],
                }
            )
    return rows


def expected_by_run() -> dict[str, dict[str, Any]]:
    expected = {}
    for row in iter_runs():
        variant = deepcopy(next(v for v in VARIANTS if v["tag"] == row["variant_tag"]))
        expected[row["run_name"]] = build_config(row["run_name"], row["arch"], variant)
    return expected


def assert_config(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)

    checks = {
        "data.reshape": config["data"].get("reshape") == "2d",
        "data.no_two_dim": "two_dim" not in config["data"],
        "data.no_log": "log" not in config["data"],
        "data.transform": config["data"].get("transform") == ["log"],
        "data.normalization": config["data"].get("normalization") == "tanh",
        "augmentations": config.get("augmentations") == expected["augmentations"],
        "model.block_out_channels": (
            config["model"]["kwargs"].get("block_out_channels")
            == expected["model"]["kwargs"]["block_out_channels"]
        ),
        "noise_scheduler.kwargs": (
            config["noise_scheduler"]["kwargs"]
            == expected["noise_scheduler"]["kwargs"]
        ),
        "train.ema_sigma_rels": config["train"].get("ema_sigma_rels") == EMA_SIGMA_RELS,
        "train.ema_burn_in": config["train"].get("ema_burn_in") == 1000,
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
    parser.add_argument("--run-name", action="append", help="Optional run name to validate/write. Repeatable.")
    parser.add_argument("--print-runs", action="store_true", help="Print selected run names and exit.")
    return parser.parse_args()


def selected_run_names(args: argparse.Namespace) -> set[str]:
    names = {row["run_name"] for row in iter_runs()}
    if args.run_name:
        names &= set(args.run_name)
    return names


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_dir = project_dir / "local" / "nf_sweep_aug" / "configs"
    manifest_path = project_dir / "local" / "nf_sweep_aug" / "manifest.json"
    expected = expected_by_run()
    selected = selected_run_names(args)

    if args.print_runs:
        for row in iter_runs():
            if row["run_name"] in selected:
                print(row["run_name"])
        return

    if not selected:
        raise SystemExit("No runs selected.")

    if args.check_only:
        for name in selected:
            assert_config(config_dir / f"{name}.yaml", expected[name])
        print(f"Validated {len(selected)} nf_sweep_aug configs.")
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
