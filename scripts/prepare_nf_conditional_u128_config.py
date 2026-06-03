#!/usr/bin/env python
"""Prepare one u128 conditional CAMELS HI run.

Default run:
- IllustrisTNG LH HI 128^3 at z=0.0
- continuous conditioning on the six CAMELS cosmology/feedback parameters
- u128 UNet2DConditionModel
- about 200k optimizer updates

The label files written here contain normalized parameter vectors.  The raw
mean/std are saved beside them so generated samples can be tied back to
physical parameter values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from prepare_nf_generalize_nick_data_configs import EMA_SIGMA_RELS, ZTHIN


SWEEP_NAME = "nf_conditional_u128"
DATA_ROOT = "/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG"
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
FIELD = "HI"
SIM = "IllustrisTNG"
DATASET = "LH"
RESOLUTION = 128
REDSHIFT = "0.0"
N_TRAIN_SIMS = 1000
TARGET_UPDATES = 200_000
CHECKPOINT_EVERY_UPDATES = 20_000
SAMPLE_N = 512
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
PARAM_EXPECTED_RANGES = {
    "Omega_m": (0.1, 0.5),
    "sigma_8": (0.6, 1.0),
    "A_SN1": (0.25, 4.0),
    "A_AGN1": (0.25, 4.0),
    "A_SN2": (0.5, 2.0),
    "A_AGN2": (0.5, 2.0),
}


def parameter_column_summary(params: np.ndarray) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for col, name in enumerate(PARAM_NAMES):
        values = params[:, col].astype(np.float64, copy=False)
        expected_min, expected_max = PARAM_EXPECTED_RANGES[name]
        observed_min = float(np.min(values))
        observed_max = float(np.max(values))
        summary.append(
            {
                "column": int(col),
                "name": name,
                "min": observed_min,
                "max": observed_max,
                "mean": float(np.mean(values)),
                "expected_min": float(expected_min),
                "expected_max": float(expected_max),
                "within_expected_range": bool(
                    observed_min >= expected_min - 5.0e-4 and observed_max <= expected_max + 5.0e-4
                ),
            }
        )
    return summary


def format_parameter_column_summary(params: np.ndarray) -> str:
    lines = [
        "CAMELS parameter column check:",
        "  assumed order: " + ", ".join(PARAM_NAMES),
        "  col  name       observed_min  observed_max  expected_range",
    ]
    for row in parameter_column_summary(params):
        lines.append(
            "  {column:>3d}  {name:<9s}  {min:>12.5g}  {max:>12.5g}  [{expected_min:g}, {expected_max:g}]".format(
                **row
            )
        )
    return "\n".join(lines)


def validate_parameter_column_ranges(params: np.ndarray, path: Path) -> None:
    bad = [row for row in parameter_column_summary(params) if not row["within_expected_range"]]
    if bad:
        detail = "\n".join(
            "  column {column} as {name}: observed [{min:.6g}, {max:.6g}], expected [{expected_min:g}, {expected_max:g}]".format(
                **row
            )
            for row in bad
        )
        raise ValueError(
            "CAMELS parameter file failed the expected column-range check. "
            f"This usually means the parameter order is wrong for {path}.\n{detail}\n"
            + format_parameter_column_summary(params)
        )


def run_name(n_train_sims: int = N_TRAIN_SIMS) -> str:
    return f"nf_cond_u128_hi_lh_z0p0_n{n_train_sims}_200k"


def image_path(data_root: str | Path) -> Path:
    return Path(data_root) / f"Grids_{FIELD}_{SIM}_{DATASET}_{RESOLUTION}_z={REDSHIFT}.npy"


def params_path(data_root: str | Path) -> Path:
    return Path(data_root) / f"params_{DATASET}_{SIM}.txt"


def steps_per_epoch(n_train_sims: int, batch_size: int) -> int:
    n_slices = int(n_train_sims) * (128 // ZTHIN)
    return max(1, math.ceil(n_slices / int(batch_size)))


def epochs_for(n_train_sims: int, batch_size: int, target_updates: int) -> int:
    return max(1, math.ceil(int(target_updates) / steps_per_epoch(n_train_sims, batch_size)))


def checkpoint_epochs_for(n_train_sims: int, batch_size: int) -> int:
    return max(1, round(CHECKPOINT_EVERY_UPDATES / steps_per_epoch(n_train_sims, batch_size)))


def load_params(path: Path, n_train_sims: int) -> np.ndarray:
    params = np.loadtxt(path, dtype=np.float32)
    if params.ndim != 2:
        raise ValueError(f"Expected 2D params table, got shape {params.shape} from {path}.")
    if params.shape[1] < len(PARAM_NAMES):
        raise ValueError(
            f"Expected at least {len(PARAM_NAMES)} parameter columns, got {params.shape[1]} from {path}."
        )
    if len(params) < n_train_sims:
        raise ValueError(f"Requested {n_train_sims} simulations but only found {len(params)} rows in {path}.")
    out = params[:n_train_sims, :len(PARAM_NAMES)].astype(np.float32, copy=False)
    validate_parameter_column_ranges(out, path)
    return out


def normalize_params(params: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mean = params.mean(axis=0)
    std = params.std(axis=0)
    std = np.where(std > 0, std, 1.0).astype(np.float32)
    normed = ((params - mean) / std).astype(np.float32)
    stats = {
        "param_names": PARAM_NAMES,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "value_ranges": parameter_column_summary(params),
        "normalization": "(raw - mean) / std",
    }
    return normed, stats


def write_labels(
    *,
    project_dir: Path,
    data_root: Path,
    n_train_sims: int,
    sample_n: int,
    check_files: bool,
) -> dict[str, Path | dict[str, Any]]:
    if check_files and not image_path(data_root).exists():
        raise FileNotFoundError(image_path(data_root))
    p_path = params_path(data_root)
    if not p_path.exists():
        raise FileNotFoundError(p_path)

    raw_params = load_params(p_path, n_train_sims)
    normed, stats = normalize_params(raw_params)

    name = run_name(n_train_sims)
    label_dir = project_dir / "local" / SWEEP_NAME / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    train_label_path = label_dir / f"{name}_train_params_norm.npy"
    train_raw_path = label_dir / f"{name}_train_params_raw.npy"
    stats_path = label_dir / f"{name}_param_norm_stats.json"
    sample_label_path = label_dir / f"{name}_sample_params_norm_n{sample_n}.npy"
    sample_raw_path = label_dir / f"{name}_sample_params_raw_n{sample_n}.npy"
    sample_index_path = label_dir / f"{name}_sample_param_indices_n{sample_n}.txt"

    sample_idx = np.linspace(0, len(normed) - 1, sample_n, dtype=np.int64)

    np.save(train_label_path, normed)
    np.save(train_raw_path, raw_params)
    np.save(sample_label_path, normed[sample_idx])
    np.save(sample_raw_path, raw_params[sample_idx])
    np.savetxt(sample_index_path, sample_idx, fmt="%d")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    return {
        "train_label_path": train_label_path,
        "train_raw_path": train_raw_path,
        "sample_label_path": sample_label_path,
        "sample_raw_path": sample_raw_path,
        "sample_index_path": sample_index_path,
        "stats_path": stats_path,
        "stats": stats,
    }


def build_config(
    *,
    project_dir: Path,
    data_root: Path,
    checkpoint_root: Path,
    n_train_sims: int,
    sample_n: int,
    label_paths: dict[str, Path | dict[str, Any]],
) -> dict[str, Any]:
    name = run_name(n_train_sims)
    batch_size = 32
    num_epochs = epochs_for(n_train_sims, batch_size, TARGET_UPDATES)
    return {
        "global": {
            "device": "cuda",
            "dtype": "float32",
        },
        "io": {
            "output_dir": str(checkpoint_root / f"{name}_checkpoints"),
        },
        "data": {
            "img_path": str(image_path(data_root)),
            "img_read_fn": "npy_read_fn",
            "label_path": str(label_paths["train_label_path"]),
            "label_read_fn": "npy_read_fn",
            "reshape": "2d",
            "zthin": ZTHIN,
            "n_samples": int(n_train_sims),
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
            "class": "UNet2DConditionModel",
            "kwargs": {
                "sample_size": 128,
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 2,
                "block_out_channels": [32, 64, 128],
                "down_block_types": [
                    "DownBlock2D",
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                ],
                "up_block_types": [
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ],
                "norm_num_groups": 32,
                "cross_attention_dim": 32,
                "encoder_hid_dim": len(PARAM_NAMES),
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
            "checkpoint_every_n_epochs": int(checkpoint_epochs_for(n_train_sims, batch_size)),
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 1,
            "dataloader_num_workers": 0,
            "max_grad_norm": 1.0,
            "conditioning": "continuous",
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
            "n_samples": int(sample_n),
            "batch_size": None,
            "image_shape": None,
            "conditioning": "continuous",
            "labels": None,
            "continuous_labels": str(label_paths["sample_label_path"]),
            "guidance_scale": None,
            "ema_sigma_rel": None,
            "seed": None,
            "device": None,
        },
    }


def manifest_row(
    *,
    project_dir: Path,
    data_root: Path,
    checkpoint_root: Path,
    n_train_sims: int,
    sample_n: int,
    label_paths: dict[str, Path | dict[str, Any]],
) -> dict[str, Any]:
    name = run_name(n_train_sims)
    batch_size = 32
    spe = steps_per_epoch(n_train_sims, batch_size)
    epochs = epochs_for(n_train_sims, batch_size, TARGET_UPDATES)
    return {
        "run_name": name,
        "arch": "u128",
        "arch_label": "UNet-128 conditional",
        "variant_tag": "continuous_cosmology_params",
        "variant_label": "u128 conditional on CAMELS params",
        "field": FIELD,
        "simulation": SIM,
        "dataset": DATASET,
        "redshift": float(REDSHIFT),
        "resolution": RESOLUTION,
        "n_train_simulations": int(n_train_sims),
        "zthin": ZTHIN,
        "slices_per_sim": 128 // ZTHIN,
        "dataset_size": int(n_train_sims * (128 // ZTHIN)),
        "steps_per_epoch": int(spe),
        "epochs": int(epochs),
        "target_updates": TARGET_UPDATES,
        "actual_updates": int(spe * epochs),
        "checkpoint_every_updates": CHECKPOINT_EVERY_UPDATES,
        "checkpoint_every_n_epochs": int(checkpoint_epochs_for(n_train_sims, batch_size)),
        "batch_size": batch_size,
        "conditioning": "continuous",
        "condition_dim": len(PARAM_NAMES),
        "param_names": PARAM_NAMES,
        "data_path": str(image_path(data_root)),
        "params_path": str(params_path(data_root)),
        "train_label_path": str(label_paths["train_label_path"]),
        "sample_label_path": str(label_paths["sample_label_path"]),
        "param_stats_path": str(label_paths["stats_path"]),
        "sample_raw_params_path": str(label_paths["sample_raw_path"]),
        "config": f"local/{SWEEP_NAME}/configs/{name}.yaml",
        "checkpoint_dir": str(checkpoint_root / f"{name}_checkpoints"),
        "sample_path": f"results/{SWEEP_NAME}/samples/{name}_seed{{seed}}_raw_conditional.npz",
        "note": "Single u128 conditional CAMELS run for testing cosmology-parameter conditioning.",
    }


def assert_config(config_path: Path, row: dict[str, Any]) -> None:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)
    labels = np.load(row["train_label_path"])
    checks = {
        "model.class": config["model"]["class"] == "UNet2DConditionModel",
        "model.cross_attention_dim": config["model"]["kwargs"].get("cross_attention_dim") == 32,
        "model.encoder_hid_dim": config["model"]["kwargs"].get("encoder_hid_dim") == len(PARAM_NAMES),
        "train.conditioning": config["train"].get("conditioning") == "continuous",
        "generate.conditioning": config["generate"].get("conditioning") == "continuous",
        "data.label_path": config["data"].get("label_path") == row["train_label_path"],
        "labels.float": np.issubdtype(labels.dtype, np.floating),
        "labels.shape": labels.shape == (row["n_train_simulations"], len(PARAM_NAMES)),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{config_path} failed checks: {', '.join(failed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="CAMELS IllustrisTNG 3d_grids directory.")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT, help="Root for saved checkpoints.")
    parser.add_argument("--n-train-sims", type=int, default=N_TRAIN_SIMS)
    parser.add_argument("--sample-n", type=int, default=SAMPLE_N)
    parser.add_argument("--check-only", action="store_true", help="Validate existing config/labels without writing.")
    parser.add_argument("--print-runs", action="store_true", help="Print run name and exit.")
    parser.add_argument("--print-table", action="store_true", help="Print one-line run table and exit.")
    parser.add_argument("--check-files", action="store_true", help="Require the image grid file to exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    name = run_name(args.n_train_sims)

    if args.print_runs:
        print(name)
        return

    config_path = project_dir / "local" / SWEEP_NAME / "configs" / f"{name}.yaml"
    manifest_path = project_dir / "local" / SWEEP_NAME / "manifest.json"

    if args.check_only:
        with manifest_path.open() as f:
            rows = json.load(f)
        if len(rows) != 1:
            raise ValueError(f"Expected one manifest row, got {len(rows)}.")
        assert_config(config_path, rows[0])
        print(f"Validated {SWEEP_NAME} config.")
        return

    label_paths = write_labels(
        project_dir=project_dir,
        data_root=data_root,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        check_files=args.check_files,
    )
    raw_params = np.load(label_paths["train_raw_path"])
    config = build_config(
        project_dir=project_dir,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        label_paths=label_paths,
    )
    row = manifest_row(
        project_dir=project_dir,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        label_paths=label_paths,
    )

    if args.print_table:
        cols = [
            "run_name",
            "dataset_size",
            "n_train_simulations",
            "steps_per_epoch",
            "epochs",
            "actual_updates",
            "conditioning",
            "condition_dim",
        ]
        print("\t".join(cols))
        print("\t".join(str(row[col]) for col in cols))
        print(format_parameter_column_summary(raw_params))
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote {config_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump([row], f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")
    print(f"Wrote labels under {project_dir / 'local' / SWEEP_NAME / 'labels'}")
    print(format_parameter_column_summary(raw_params))


if __name__ == "__main__":
    main()
