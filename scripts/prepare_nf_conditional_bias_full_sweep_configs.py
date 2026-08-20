#!/usr/bin/env python
"""Prepare ten freshly initialized conditional UNet calibration runs.

Every power-of-two field count from 2^6 through 2^15 is trained independently
to the same 200k-update target.  The validated heldout split, normalization
statistics, and parameter statistics remain fixed across the sweep; no trained
generator checkpoint is reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import prepare_nf_conditional_bias_probe_configs as base
from prepare_nf_conditional_u128_config import PARAM_NAMES, image_path, load_params, params_path


SWEEP_NAME = "nf_conditional_bias_fresh_full_sweep_200k"
SHARED_STATE_SWEEP_NAME = base.SWEEP_NAME
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
PREPARED_DATA_ROOT = f"{CHECKPOINT_ROOT}/prepared_data"
ALL_DATASET_SIZES = tuple(2**p for p in range(6, 16))
REUSED_DATASET_SIZES: tuple[int, ...] = ()
TRAIN_DATASET_SIZES = ALL_DATASET_SIZES
TRAINING_SEED = 123


def run_name(dataset_size: int) -> str:
    exponent = int(round(math.log2(int(dataset_size))))
    if 2**exponent != int(dataset_size):
        raise ValueError(f"dataset_size must be a power of two, got {dataset_size}")
    return f"nf_cond_bias_hi_u128_d2p{exponent:02d}_n{int(dataset_size)}_fresh200k_fullsweep"


def checkpoint_epoch_for(dataset_size: int) -> int:
    """Return the exact final epoch corresponding to the fixed 200k target."""

    return base.epochs_for(int(dataset_size), base.TARGET_UPDATES) - 1


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def scientific_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Return every setting that can change the trained or sampled distribution."""

    data = config["data"]
    train = config["train"]
    generate = config["generate"]
    signature = {
        "data": {
            "reshape": data.get("reshape"),
            "zthin": data.get("zthin"),
            "normalization": data.get("normalization"),
            "norm_kwargs": data.get("norm_kwargs"),
            "transform": data.get("transform"),
        },
        "model": config["model"],
        "noise_scheduler": config["noise_scheduler"],
        "optimizer": config["optimizer"],
        "lr_scheduler": config["lr_scheduler"],
        "train": {
            key: train.get(key)
            for key in (
                "num_epochs",
                "batch_size",
                "shuffle",
                "mixed_precision",
                "gradient_accumulation_steps",
                "max_grad_norm",
                "conditioning",
                "cfg_dropout",
                "ema_sigma_rels",
                "ema_update_every",
                "ema_burn_in",
                "min_snr_gamma",
                "sigma_log_normal",
            )
        },
        "generate": {
            key: generate.get(key)
            for key in (
                "scheduler",
                "num_steps",
                "conditioning",
                "guidance_scale",
                "ema_sigma_rel",
            )
        },
    }
    return _canonical(signature)


def assert_matching_scientific_protocol(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    reference_signature = scientific_signature(reference)
    candidate_signature = scientific_signature(candidate)
    if reference_signature != candidate_signature:
        differing = [
            key
            for key in reference_signature
            if reference_signature.get(key) != candidate_signature.get(key)
        ]
        raise ValueError(
            "Cannot reuse checkpoint: scientific protocol differs in "
            + ", ".join(differing)
        )


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    invalid = sorted(set(sizes) - set(ALL_DATASET_SIZES))
    if invalid:
        raise ValueError(f"Unsupported dataset sizes: {invalid}")
    return sizes


def nested_slice_pairs_by_size(
    allowed_sims: np.ndarray,
    dataset_sizes: tuple[int, ...],
    z_size: int,
) -> dict[int, np.ndarray]:
    """Return deterministic, evenly spread, nested training subsets."""

    sizes = tuple(sorted({int(size) for size in dataset_sizes}))
    if not sizes or any(size <= 0 or size & (size - 1) for size in sizes):
        raise ValueError("dataset sizes must be positive powers of two")
    maximum = sizes[-1]
    pool = base.select_slice_pairs(allowed_sims, maximum, z_size)
    bit_width = int(math.log2(maximum))
    indices = np.arange(maximum, dtype=np.uint64)
    reversed_indices = np.zeros(maximum, dtype=np.uint64)
    for bit in range(bit_width):
        reversed_indices |= ((indices >> bit) & 1) << (bit_width - bit - 1)
    ordered_pool = pool[reversed_indices.astype(np.int64)]
    return {size: ordered_pool[:size].copy() for size in sizes}


def _load_shared_state(project_dir: Path) -> tuple[dict[str, float], dict[str, Any], np.ndarray, Path, Path, Path]:
    heldout = project_dir / "local" / SHARED_STATE_SWEEP_NAME / "heldout"
    norm_path = heldout / "shared_image_norm_stats.json"
    param_stats_path = heldout / "param_norm_stats.json"
    heldout_indices_path = heldout / "heldout_simulation_indices.txt"
    heldout_norm_path = heldout / f"heldout_params_norm_k{base.SAMPLE_K_PER_COSMOLOGY}.npy"
    heldout_raw_path = heldout / "heldout_params_raw.npy"
    required = [norm_path, param_stats_path, heldout_indices_path, heldout_norm_path, heldout_raw_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing validated shared conditional artifacts: " + ", ".join(missing))
    norm_info = json.loads(norm_path.read_text())
    param_stats = json.loads(param_stats_path.read_text())
    heldout_indices = np.atleast_1d(np.loadtxt(heldout_indices_path, dtype=np.int64))
    expected = np.arange(base.HELDOUT_START, base.HELDOUT_START + base.HELDOUT_COUNT)
    if not np.array_equal(heldout_indices, expected):
        raise ValueError(f"Heldout simulations must be 900-931, found {heldout_indices.tolist()}")
    return norm_info, param_stats, heldout_indices, heldout_indices_path, heldout_norm_path, heldout_raw_path


def _write_pairs(path: Path, pairs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "simulation_index", "z_index"])
        for index, (simulation, z_index) in enumerate(pairs):
            writer.writerow([index, int(simulation), int(z_index)])


def _new_manifest_row(
    *,
    dataset_size: int,
    config_path: Path,
    config: dict[str, Any],
    image_path_value: Path,
    label_path: Path,
    raw_label_path: Path,
    pairs_path: Path,
    heldout_indices_path: Path,
    heldout_norm_path: Path,
    heldout_raw_path: Path,
    norm_info: dict[str, Any],
) -> dict[str, Any]:
    steps = base.steps_per_epoch(dataset_size)
    epochs = base.epochs_for(dataset_size, base.TARGET_UPDATES)
    checkpoint_epoch = checkpoint_epoch_for(dataset_size)
    name = run_name(dataset_size)
    return {
        "run_name": name,
        "source_run_name": None,
        "reuse_existing_checkpoint": False,
        "initialization": "fresh",
        "training_seed": TRAINING_SEED,
        "regime": "full_sweep",
        "arch": "u128",
        "dataset_size": int(dataset_size),
        "steps_per_epoch": int(steps),
        "epochs": int(epochs),
        "target_updates": int(base.TARGET_UPDATES),
        "actual_updates": int(steps * epochs),
        "batch_size": int(base.BATCH_SIZE),
        "conditioning": "continuous",
        "condition_dim": len(PARAM_NAMES),
        "param_names": list(PARAM_NAMES),
        "prepared_image_path": str(image_path_value),
        "train_label_path": str(label_path),
        "train_raw_params_path": str(raw_label_path),
        "selected_pairs_path": str(pairs_path),
        "heldout_indices_path": str(heldout_indices_path),
        "heldout_sample_params_norm_path": str(heldout_norm_path),
        "heldout_raw_params_path": str(heldout_raw_path),
        "heldout_samples_per_cosmology": int(base.SAMPLE_K_PER_COSMOLOGY),
        "normalization": {"transform": ["log"], "method": "tanh", **norm_info},
        "config": str(config_path.relative_to(config_path.parents[3])),
        "checkpoint_dir": str(config["io"]["output_dir"]),
        "checkpoint_epoch": int(checkpoint_epoch),
        "requested_checkpoint": str(
            Path(config["io"]["output_dir"]) / f"checkpoint-epoch-{checkpoint_epoch:04d}"
        ),
        "sample_path": f"results/{SWEEP_NAME}/samples/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
        "note": "Full six-parameter conditional calibration sweep; heldout CAMELS simulations 900-931.",
    }


def prepare(project_dir: Path, data_root: Path, checkpoint_root: Path, prepared_root: Path, write_arrays: bool) -> list[dict[str, Any]]:
    norm_info, param_stats, heldout, heldout_indices_path, heldout_norm_path, heldout_raw_path = _load_shared_state(project_dir)

    raw_params = load_params(params_path(data_root), 1000)
    allowed = base.allowed_indices(len(raw_params), heldout)
    mean = np.asarray(param_stats["mean"], dtype=np.float32)
    std = np.asarray(param_stats["std"], dtype=np.float32)
    normalized_params = ((raw_params - mean) / std).astype(np.float32)
    grid_path = image_path(data_root)
    selected_pairs = nested_slice_pairs_by_size(allowed, ALL_DATASET_SIZES, z_size=128)

    local_root = project_dir / "local" / SWEEP_NAME
    config_dir = local_root / "configs"
    label_dir = local_root / "labels"
    config_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset_size in ALL_DATASET_SIZES:
        name = run_name(dataset_size)
        config_path = config_dir / f"{name}.yaml"
        pairs = selected_pairs[dataset_size]
        pairs_file = label_dir / f"{name}_selected_slices.csv"
        image_file = prepared_root / f"{name}_train_images.npy"
        label_file = label_dir / f"{name}_train_params_norm.npy"
        raw_label_file = label_dir / f"{name}_train_params_raw.npy"
        _write_pairs(pairs_file, pairs)
        if write_arrays:
            base.materialize_slices(grid_path=grid_path, out_path=image_file, pairs=pairs)
            np.save(label_file, normalized_params[pairs[:, 0]])
            np.save(raw_label_file, raw_params[pairs[:, 0]])
        config = base.build_config(
            checkpoint_root=checkpoint_root,
            prepared_data_root=prepared_root,
            dataset_size=dataset_size,
            norm_info=norm_info,
            image_file=image_file,
            label_file=label_file,
            heldout_label_file=heldout_norm_path,
            heldout_count=len(heldout),
            sample_k=base.SAMPLE_K_PER_COSMOLOGY,
        )
        config["io"]["output_dir"] = str(checkpoint_root / f"{name}_checkpoints")

        config["generate"]["continuous_labels"] = str(heldout_norm_path)
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        rows.append(
            _new_manifest_row(
                dataset_size=dataset_size,
                config_path=config_path,
                config=config,
                image_path_value=image_file,
                label_path=label_file,
                raw_label_path=raw_label_file,
                pairs_path=pairs_file,
                heldout_indices_path=heldout_indices_path,
                heldout_norm_path=heldout_norm_path,
                heldout_raw_path=heldout_raw_path,
                norm_info=norm_info,
            )
        )
        print(f"Wrote {config_path}")

    manifest_path = local_root / "manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--data-root", default=base.DATA_ROOT)
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--prepared-data-root", default=PREPARED_DATA_ROOT)
    parser.add_argument("--write-arrays", action="store_true")
    parser.add_argument("--print-runs", action="store_true")
    parser.add_argument("--print-train-runs", action="store_true")
    args = parser.parse_args()

    if args.print_runs:
        for size in ALL_DATASET_SIZES:
            print(run_name(size))
        return
    if args.print_train_runs:
        for size in TRAIN_DATASET_SIZES:
            print(run_name(size))
        return

    rows = prepare(
        Path(args.project_dir).resolve(),
        Path(args.data_root),
        Path(args.checkpoint_root),
        Path(args.prepared_data_root),
        args.write_arrays,
    )
    print("Prepared dataset sizes:", ",".join(str(row["dataset_size"]) for row in rows))
    print("Reused generator checkpoints: none")
    print("Fresh training tasks:", len(TRAIN_DATASET_SIZES))
    print("Training RNG seed:", TRAINING_SEED)


if __name__ == "__main__":
    main()
