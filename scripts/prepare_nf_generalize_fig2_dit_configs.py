#!/usr/bin/env python
"""Prepare one-architecture DiT Fig.2-style generalization configs.

This is the transformer analogue of ``prepare_nf_generalize_fig2_configs.py``:
one DiT architecture is trained across the same CAMELS HI training-set sizes so
its memorization-to-generalization transition can be measured with the existing
PCA/SSCD nearest-neighbor diagnostics.

The sweep intentionally lives under a separate name so the DiT samples and
tables do not overwrite the UNet Fig.2 results.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from prepare_nf_generalize_fig2_configs import (
    CHECKPOINT_EVERY_UPDATES,
    GENERATE_N_SAMPLES,
    RUN_SIZES,
    TARGET_UPDATES,
)
from prepare_nf_generalize_nick_data_configs import (
    DATA_SOURCES,
    EMA_SIGMA_RELS,
    SLICES_PER_VOLUME,
    ZTHIN,
    allocate_source_counts,
)


SWEEP_NAME = "nf_generalize_fig2_dit"
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"

DIT_VARIANTS: dict[str, dict[str, Any]] = {
    "dit_base": {
        "label": "DiT-base",
        "variant_tag": "dit_base_noaug_fixed_updates",
        "variant_label": "DiT-base, no augmentation, fixed 200k updates",
        # Matches the existing template: roughly U128-size parameter count.
        "model_kwargs": {
            "sample_size": 128,
            "patch_size": 8,
            "in_channels": 1,
            "out_channels": 1,
            "num_layers": 12,
            "num_attention_heads": 12,
            "attention_head_dim": 64,
            "num_embeds_ada_norm": 1,
            "norm_num_groups": 32,
        },
        # Keep micro-batches small for DiT memory.  Epoch counts below are based
        # on optimizer updates after gradient accumulation.
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
    },
}


def run_name(variant: str, dataset_tag: str) -> str:
    return f"nf_fig2_{variant}_{dataset_tag}_noaug_200k"


def micro_steps_per_epoch(target_2d: int, batch_size: int) -> int:
    return max(1, math.ceil(int(target_2d) / int(batch_size)))


def optimizer_steps_per_epoch(target_2d: int, batch_size: int, grad_accum: int) -> int:
    return max(1, math.ceil(micro_steps_per_epoch(target_2d, batch_size) / int(grad_accum)))


def epochs_for(target_2d: int, batch_size: int, grad_accum: int) -> int:
    return max(1, math.ceil(TARGET_UPDATES / optimizer_steps_per_epoch(target_2d, batch_size, grad_accum)))


def checkpoint_epochs_for(target_2d: int, batch_size: int, grad_accum: int) -> int:
    return max(1, round(CHECKPOINT_EVERY_UPDATES / optimizer_steps_per_epoch(target_2d, batch_size, grad_accum)))


def build_config(name: str, variant: str, sources: list[dict[str, Any]], target_2d: int) -> dict[str, Any]:
    variant_cfg = DIT_VARIANTS[variant]
    batch_size = int(variant_cfg["batch_size"])
    grad_accum = int(variant_cfg["gradient_accumulation_steps"])
    num_epochs = epochs_for(target_2d, batch_size, grad_accum)
    checkpoint_every_n_epochs = checkpoint_epochs_for(target_2d, batch_size, grad_accum)

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
            # diffusers DiTTransformer2DModel requires class_labels for adaLN.
            # For an unconditional run, every image gets the same null class.
            "constant_label": 0,
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
            "class": "DiTTransformer2DModel",
            "kwargs": deepcopy(variant_cfg["model_kwargs"]),
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
            "gradient_accumulation_steps": grad_accum,
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
            "labels": [0],
            "continuous_labels": None,
            "guidance_scale": None,
            "ema_sigma_rel": None,
            "seed": None,
            "device": None,
        },
    }


def iter_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, variant_cfg in DIT_VARIANTS.items():
        for dataset_tag, target_2d in RUN_SIZES:
            name = run_name(variant, dataset_tag)
            sources = allocate_source_counts(target_2d)
            batch_size = int(variant_cfg["batch_size"])
            grad_accum = int(variant_cfg["gradient_accumulation_steps"])
            micro_spe = micro_steps_per_epoch(target_2d, batch_size)
            opt_spe = optimizer_steps_per_epoch(target_2d, batch_size, grad_accum)
            epochs = epochs_for(target_2d, batch_size, grad_accum)
            actual_updates = opt_spe * epochs
            rows.append(
                {
                    "run_name": name,
                    "arch": variant,
                    "arch_label": variant_cfg["label"],
                    "variant_tag": variant_cfg["variant_tag"],
                    "variant_label": variant_cfg["variant_label"],
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
                    "micro_steps_per_epoch": int(micro_spe),
                    "steps_per_epoch": int(opt_spe),
                    "optimizer_steps_per_epoch": int(opt_spe),
                    "target_updates": TARGET_UPDATES,
                    "actual_updates": int(actual_updates),
                    "checkpoint_every_updates": CHECKPOINT_EVERY_UPDATES,
                    "checkpoint_every_n_epochs": int(checkpoint_epochs_for(target_2d, batch_size, grad_accum)),
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "effective_batch_size": batch_size * grad_accum,
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
                    "sample_path": f"results/{SWEEP_NAME}/samples/{name}_seed{{seed}}_{{sample_label}}.npz",
                    "note": "DiT Fig.2-style run: no augmentation and about 200k optimizer updates.",
                }
            )
    return rows


def expected_by_run() -> dict[str, dict[str, Any]]:
    out = {}
    for row in iter_runs():
        out[row["run_name"]] = build_config(
            row["run_name"],
            row["arch"],
            deepcopy(row["source_counts"]),
            int(row["dataset_size"]),
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
        "data.constant_label": config["data"].get("constant_label") == 0,
        "data.zthin": config["data"].get("zthin") == ZTHIN,
        "data.transform": config["data"].get("transform") == ["log"],
        "data.normalization": config["data"].get("normalization") == "tanh",
        "data.target_size": sum(config["data"]["n_samples"]) * SLICES_PER_VOLUME == row["dataset_size"],
        "augmentations.absent": "augmentations" not in config,
        "model.class": config["model"].get("class") == "DiTTransformer2DModel",
        "model.kwargs": config["model"]["kwargs"] == expected["model"]["kwargs"],
        "noise_scheduler.kwargs": config["noise_scheduler"]["kwargs"] == expected["noise_scheduler"]["kwargs"],
        "train.num_epochs": config["train"].get("num_epochs") == row["epochs"],
        "train.batch_size": config["train"].get("batch_size") == row["batch_size"],
        "train.gradient_accumulation_steps": (
            config["train"].get("gradient_accumulation_steps") == row["gradient_accumulation_steps"]
        ),
        "train.conditioning": config["train"].get("conditioning") == "discrete",
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
    if args.variant:
        wanted = set(args.variant)
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
    parser.add_argument(
        "--variant",
        choices=sorted(DIT_VARIANTS),
        action="append",
        help="Restrict to a DiT variant. Repeatable.",
    )
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
