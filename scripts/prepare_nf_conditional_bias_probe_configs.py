#!/usr/bin/env python
"""Prepare continuous-cosmology HI bias-probe configs.

This experiment compares two continuous-conditioning regimes with the same
u128 UNet2DConditionModel architecture:

* N=128 selected HI 2D training fields (memorization regime)
* N=16384 selected HI 2D training fields (generalization regime)

Unlike ``prepare_nf_conditional_u128_config.py``, ``N`` here is the number of
2D training fields, not the number of CAMELS simulation cubes.  The selected
2D fields and matching parameter labels are materialized once so the cosmodiff
loader can consume exact field counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from prepare_nf_conditional_u128_config import (
    DATA_ROOT,
    DATASET,
    FIELD,
    PARAM_NAMES,
    format_parameter_column_summary,
    REDSHIFT,
    RESOLUTION,
    SIM,
    image_path,
    load_params,
    normalize_params,
    params_path,
)
from prepare_nf_generalize_nick_data_configs import EMA_SIGMA_RELS


SWEEP_NAME = "nf_conditional_bias_probe"
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
PREPARED_DATA_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}/prepared_data"
DATASET_SIZES = (128, 16_384)
TARGET_UPDATES = 200_000
CHECKPOINT_EVERY_UPDATES = 20_000
BATCH_SIZE = 32
HELDOUT_COUNT = 32
HELDOUT_START = 900
NORM_FIT_SLICES = 4096
SAMPLE_K_PER_COSMOLOGY = 64


def run_name(dataset_size: int) -> str:
    exp = int(round(math.log2(int(dataset_size))))
    if 2**exp != int(dataset_size):
        raise ValueError(f"dataset_size must be a power of two, got {dataset_size}.")
    return f"nf_cond_bias_hi_u128_d2p{exp:02d}_n{int(dataset_size)}_200k"


def steps_per_epoch(dataset_size: int, batch_size: int = BATCH_SIZE) -> int:
    return max(1, math.ceil(int(dataset_size) / int(batch_size)))


def epochs_for(dataset_size: int, target_updates: int, batch_size: int = BATCH_SIZE) -> int:
    return max(1, math.ceil(int(target_updates) / steps_per_epoch(dataset_size, batch_size)))


def checkpoint_epochs_for(dataset_size: int, batch_size: int = BATCH_SIZE) -> int:
    return max(1, round(CHECKPOINT_EVERY_UPDATES / steps_per_epoch(dataset_size, batch_size)))


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def heldout_indices(n_sims: int, start: int, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("heldout count must be positive.")
    if start < 0 or start + count > n_sims:
        raise ValueError(f"heldout range [{start}, {start + count}) is outside 0..{n_sims}.")
    return np.arange(start, start + count, dtype=np.int64)


def allowed_indices(n_sims: int, heldout: np.ndarray) -> np.ndarray:
    mask = np.ones(n_sims, dtype=bool)
    mask[heldout] = False
    return np.flatnonzero(mask).astype(np.int64)


def select_slice_pairs(allowed_sims: np.ndarray, dataset_size: int, z_size: int) -> np.ndarray:
    """Select exact ``dataset_size`` unique ``(sim, z)`` pairs spread over sims."""

    total = int(len(allowed_sims)) * int(z_size)
    if dataset_size > total:
        raise ValueError(f"Cannot select {dataset_size} slices from {total} available non-heldout slices.")
    flat_idx = np.linspace(0, total - 1, int(dataset_size), dtype=np.int64)
    sims = allowed_sims[flat_idx // int(z_size)]
    z = flat_idx % int(z_size)
    return np.stack([sims, z.astype(np.int64)], axis=1)


def select_norm_fit_pairs(allowed_sims: np.ndarray, z_size: int, max_slices: int) -> np.ndarray:
    n = min(int(max_slices), int(len(allowed_sims)) * int(z_size))
    return select_slice_pairs(allowed_sims, n, z_size)


def materialize_slices(
    *,
    grid_path: Path,
    out_path: Path,
    pairs: np.ndarray,
) -> None:
    arr = np.load(grid_path, mmap_mode="r")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(pairs), 1, arr.shape[-2], arr.shape[-1]),
    )
    for row, (sim_idx, z_idx) in enumerate(pairs):
        out[row, 0] = np.asarray(arr[int(sim_idx), int(z_idx)], dtype=np.float32)
    out.flush()


def compute_log_center_xmax(grid_path: Path, pairs: np.ndarray) -> dict[str, float]:
    arr = np.load(grid_path, mmap_mode="r")
    total = 0
    sum_log = 0.0
    for sim_idx, z_idx in pairs:
        img = np.asarray(arr[int(sim_idx), int(z_idx)], dtype=np.float32)
        logged = np.log(np.maximum(img, np.float32(1.0e-30))).astype(np.float32, copy=False)
        sum_log += float(np.sum(logged, dtype=np.float64))
        total += int(logged.size)
    center = sum_log / max(total, 1)

    xmax = 0.0
    for sim_idx, z_idx in pairs:
        img = np.asarray(arr[int(sim_idx), int(z_idx)], dtype=np.float32)
        logged = np.log(np.maximum(img, np.float32(1.0e-30))).astype(np.float32, copy=False)
        xmax = max(xmax, float(np.max(np.abs(logged - np.float32(center)))))
    return {
        "center": float(center),
        "xmax": float(max(xmax, 1.0e-30)),
        "norm_fit_slices": int(len(pairs)),
    }


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def write_pairs_csv(path: Path, pairs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "simulation_index", "z_index"])
        for row, (sim_idx, z_idx) in enumerate(pairs):
            writer.writerow([row, int(sim_idx), int(z_idx)])


def build_config(
    *,
    checkpoint_root: Path,
    prepared_data_root: Path,
    dataset_size: int,
    norm_info: dict[str, float],
    image_file: Path,
    label_file: Path,
    heldout_label_file: Path,
    heldout_count: int,
    sample_k: int,
) -> dict[str, Any]:
    name = run_name(dataset_size)
    num_epochs = epochs_for(dataset_size, TARGET_UPDATES)
    norm_kwargs = {
        "center": float(norm_info["center"]),
        "xmax": float(norm_info["xmax"]),
        "alpha": 0.8,
        "beta": 10.0,
        "delta": 1.0,
        "gamma": 1.0,
        "sigma": 1.5,
    }
    return {
        "global": {"device": "cuda", "dtype": "float32"},
        "io": {"output_dir": str(checkpoint_root / f"{name}_checkpoints")},
        "data": {
            "img_path": str(image_file),
            "img_read_fn": "npy_read_fn",
            "label_path": str(label_file),
            "label_read_fn": "npy_read_fn",
            "reshape": None,
            "zthin": 1,
            "n_samples": None,
            "seed": None,
            "keep_on_cpu": True,
            "normalization": "tanh",
            "norm_kwargs": norm_kwargs,
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
                "down_block_types": ["DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D"],
                "up_block_types": ["CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
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
        "optimizer": {"class": "AdamW", "kwargs": {"lr": 1.0e-4, "weight_decay": 1.0e-2}},
        "lr_scheduler": {"class": "CosineAnnealingWarmRestarts", "kwargs": {"T_0": 4000, "eta_min": 1.0e-7}},
        "train": {
            "num_epochs": int(num_epochs),
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "checkpoint_every_n_epochs": int(checkpoint_epochs_for(dataset_size)),
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
            "scheduler": "DPMSolverMultistepScheduler",
            "num_steps": 50,
            "s_churn": None,
            "s_tmin": None,
            "s_tmax": None,
            "s_noise": None,
            "n_samples": int(sample_k) * int(heldout_count),
            "batch_size": 8,
            "image_shape": None,
            "conditioning": "continuous",
            "labels": None,
            "continuous_labels": str(heldout_label_file),
                "guidance_scale": None,
            "ema_sigma_rel": None,
            "seed": None,
            "device": None,
        },
    }


def manifest_row(
    *,
    project_dir: Path,
    checkpoint_root: Path,
    prepared_data_root: Path,
    dataset_size: int,
    pairs_file: Path,
    image_file: Path,
    label_file: Path,
    raw_label_file: Path,
    norm_info: dict[str, float],
    heldout_file: Path,
    heldout_label_file: Path,
    sample_k: int,
) -> dict[str, Any]:
    name = run_name(dataset_size)
    spe = steps_per_epoch(dataset_size)
    epochs = epochs_for(dataset_size, TARGET_UPDATES)
    return {
        "run_name": name,
        "regime": "memorization" if int(dataset_size) <= 128 else "generalization",
        "arch": "u128",
        "field": FIELD,
        "simulation": SIM,
        "dataset": DATASET,
        "redshift": float(REDSHIFT),
        "resolution": RESOLUTION,
        "dataset_size": int(dataset_size),
        "steps_per_epoch": int(spe),
        "epochs": int(epochs),
        "target_updates": int(TARGET_UPDATES),
        "actual_updates": int(spe * epochs),
        "checkpoint_every_updates": int(CHECKPOINT_EVERY_UPDATES),
        "checkpoint_every_n_epochs": int(checkpoint_epochs_for(dataset_size)),
        "batch_size": int(BATCH_SIZE),
        "conditioning": "continuous",
        "cfg_dropout": 0.0,
        "guidance_scale": None,
        "condition_dim": len(PARAM_NAMES),
        "param_names": PARAM_NAMES,
        "data_path": str(image_path(DATA_ROOT)),
        "params_path": str(params_path(DATA_ROOT)),
        "prepared_image_path": str(image_file),
        "prepared_image_sha256": file_sha256(image_file) if image_file.exists() else None,
        "train_label_path": str(label_file),
        "train_raw_params_path": str(raw_label_file),
        "selected_pairs_path": str(pairs_file),
        "heldout_indices_path": str(heldout_file),
        "heldout_sample_params_norm_path": str(heldout_label_file),
        "heldout_samples_per_cosmology": int(sample_k),
        "normalization": {
            "transform": ["log"],
            "method": "tanh",
            **norm_info,
        },
        "config": f"local/{SWEEP_NAME}/configs/{name}.yaml",
        "checkpoint_dir": str(checkpoint_root / f"{name}_checkpoints"),
        "sample_path": f"results/{SWEEP_NAME}/samples/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
        "note": (
            "Continuous HI cosmology bias probe. Exact N counts materialized 2D fields; "
            "heldout cosmologies excluded from diffusion and encoder training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--prepared-data-root", default=PREPARED_DATA_ROOT)
    parser.add_argument("--dataset-sizes", default=",".join(str(x) for x in DATASET_SIZES))
    parser.add_argument("--heldout-start", type=int, default=HELDOUT_START)
    parser.add_argument("--heldout-count", type=int, default=HELDOUT_COUNT)
    parser.add_argument("--norm-fit-slices", type=int, default=NORM_FIT_SLICES)
    parser.add_argument("--sample-k-per-cosmology", type=int, default=SAMPLE_K_PER_COSMOLOGY)
    parser.add_argument("--write-arrays", action="store_true", help="Materialize selected 2D slices and labels.")
    parser.add_argument("--print-runs", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    prepared_data_root = Path(args.prepared_data_root)
    dataset_sizes = parse_int_list(args.dataset_sizes)

    if args.print_runs:
        for n in dataset_sizes:
            print(run_name(n))
        return

    grid_path = image_path(data_root)
    raw_params_all = load_params(params_path(data_root), 1000)
    n_sims = len(raw_params_all)
    heldout = heldout_indices(n_sims, args.heldout_start, args.heldout_count)
    allowed = allowed_indices(n_sims, heldout)
    normed_params_all, param_stats = normalize_params(raw_params_all[allowed])
    param_mean = np.asarray(param_stats["mean"], dtype=np.float32)
    param_std = np.asarray(param_stats["std"], dtype=np.float32)
    params_norm_all = ((raw_params_all - param_mean) / param_std).astype(np.float32)

    heldout_dir = project_dir / "local" / SWEEP_NAME / "heldout"
    label_dir = project_dir / "local" / SWEEP_NAME / "labels"
    config_dir = project_dir / "local" / SWEEP_NAME / "configs"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    heldout_index_path = heldout_dir / "heldout_simulation_indices.txt"
    np.savetxt(heldout_index_path, heldout, fmt="%d")
    np.save(heldout_dir / "heldout_params_raw.npy", raw_params_all[heldout])
    np.save(heldout_dir / "heldout_params_norm.npy", params_norm_all[heldout])
    heldout_repeated_path = heldout_dir / f"heldout_params_norm_k{args.sample_k_per_cosmology}.npy"
    np.save(heldout_repeated_path, np.repeat(params_norm_all[heldout], args.sample_k_per_cosmology, axis=0))
    (heldout_dir / "param_norm_stats.json").write_text(json.dumps(param_stats, indent=2) + "\n")

    if args.write_arrays:
        norm_pairs = select_norm_fit_pairs(allowed, 128, args.norm_fit_slices)
        norm_info = compute_log_center_xmax(grid_path, norm_pairs)
    else:
        norm_info = {
            "center": None,
            "xmax": None,
            "norm_fit_slices": int(args.norm_fit_slices),
        }
    norm_info_path = heldout_dir / "shared_image_norm_stats.json"
    if args.write_arrays:
        norm_info_path.write_text(json.dumps(norm_info, indent=2) + "\n")
    elif norm_info_path.exists():
        norm_info = json.loads(norm_info_path.read_text())
    else:
        raise FileNotFoundError(
            f"Missing {norm_info_path}. Run once with --write-arrays on Great Lakes to fit shared normalization."
        )

    rows: list[dict[str, Any]] = []
    for dataset_size in dataset_sizes:
        name = run_name(dataset_size)
        pairs = select_slice_pairs(allowed, dataset_size, 128)
        pairs_path = label_dir / f"{name}_selected_slices.csv"
        image_file = prepared_data_root / f"{name}_train_images.npy"
        label_file = label_dir / f"{name}_train_params_norm.npy"
        raw_label_file = label_dir / f"{name}_train_params_raw.npy"

        if args.write_arrays:
            write_pairs_csv(pairs_path, pairs)
            materialize_slices(grid_path=grid_path, out_path=image_file, pairs=pairs)
            np.save(label_file, params_norm_all[pairs[:, 0]])
            np.save(raw_label_file, raw_params_all[pairs[:, 0]])
        elif not pairs_path.exists():
            write_pairs_csv(pairs_path, pairs)

        config = build_config(
            checkpoint_root=checkpoint_root,
            prepared_data_root=prepared_data_root,
            dataset_size=dataset_size,
            norm_info=norm_info,
            image_file=image_file,
            label_file=label_file,
            heldout_label_file=heldout_repeated_path,
            heldout_count=len(heldout),
            sample_k=args.sample_k_per_cosmology,
        )
        config_path = config_dir / f"{name}.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        rows.append(
            manifest_row(
                project_dir=project_dir,
                checkpoint_root=checkpoint_root,
                prepared_data_root=prepared_data_root,
                dataset_size=dataset_size,
                pairs_file=pairs_path,
                image_file=image_file,
                label_file=label_file,
                raw_label_file=raw_label_file,
                norm_info=norm_info,
                heldout_file=heldout_index_path,
                heldout_label_file=heldout_repeated_path,
                sample_k=args.sample_k_per_cosmology,
            )
        )
        print(f"Wrote {config_path}")

    manifest_path = project_dir / "local" / SWEEP_NAME / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    print("heldout simulations:", ",".join(str(int(x)) for x in heldout))
    print("parameter order:", ",".join(PARAM_NAMES))
    print(format_parameter_column_summary(raw_params_all))

    if args.print_table:
        cols = ["run_name", "regime", "dataset_size", "steps_per_epoch", "epochs", "actual_updates", "checkpoint_every_n_epochs"]
        print("\t".join(cols))
        for row in rows:
            print("\t".join(str(row[col]) for col in cols))


if __name__ == "__main__":
    main()
