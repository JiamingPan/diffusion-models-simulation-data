#!/usr/bin/env python
"""Prepare the fixed-cosmology, varying-multiplicity conditional sweep.

Four freshly initialized conditional UNet runs that share the SAME 64 training
simulations as the existing N=64 full-sweep run and differ only in how many
z-slices each simulation contributes:

    m = 1, 2, 4, 8   ->   N = 64, 128, 256, 512

The m=1 training set is byte-for-byte the existing N=64 selection (same (sim, z)
rows in the same order). For m > 1 each simulation's slices are
z_j = (z_0 + j*128/m) mod 128, j = 0..m-1, so the sets are nested and evenly
spaced around the periodic box. Everything else (architecture, scheduler,
200k-update budget, normalization, held-out split, sampling protocol) is
inherited unchanged from the full sweep via its config builder.

Pre-registered prediction (write before running): m=1 reproduces the existing
N=64 behaviour (draw-to-draw cosine ~1, |Omega_m bias| ~0.15); by m=4-8 draw
diversity reaches the real-realization level and coverage recovers, at N where
the one-map-per-label sweep is still collapsed.
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
import prepare_nf_conditional_bias_full_sweep_configs as full
from prepare_nf_conditional_u128_config import PARAM_NAMES, image_path, load_params, params_path

SWEEP_NAME = "nf_conditional_fixedC64_multiplicity_200k"
SOURCE_SWEEP_NAME = full.SWEEP_NAME
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
PREPARED_DATA_ROOT = f"{CHECKPOINT_ROOT}/prepared_data"
MULTIPLICITIES = (1, 2, 4, 8)
N_COSMOLOGIES = 64
Z_SIZE = 128
TRAINING_SEED = full.TRAINING_SEED


def run_name(multiplicity: int) -> str:
    if multiplicity not in MULTIPLICITIES:
        raise ValueError(f"multiplicity must be one of {MULTIPLICITIES}, got {multiplicity}")
    return f"nf_cond_fixedC64_m{multiplicity:02d}_n{N_COSMOLOGIES * multiplicity}_fresh200k"


def dataset_size(multiplicity: int) -> int:
    return N_COSMOLOGIES * int(multiplicity)


def load_source_pairs(project_dir: Path) -> np.ndarray:
    """The exact (sim, z) rows of the existing N=64 run, in training-row order."""
    path = (project_dir / "local" / SOURCE_SWEEP_NAME / "labels"
            / f"{full.run_name(N_COSMOLOGIES)}_selected_slices.csv")
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    pairs = np.array([[int(r["simulation_index"]), int(r["z_index"])] for r in rows], dtype=np.int64)
    order = np.array([int(r["row"]) for r in rows])
    if len(pairs) != N_COSMOLOGIES or not np.array_equal(order, np.arange(N_COSMOLOGIES)):
        raise ValueError(f"source N=64 selection is not 64 ordered rows: {path}")
    if len(np.unique(pairs[:, 0])) != N_COSMOLOGIES:
        raise ValueError("source N=64 selection must have one slice per simulation")
    return pairs


def multiplicity_pairs(source_pairs: np.ndarray, multiplicity: int, z_size: int = Z_SIZE) -> np.ndarray:
    """Cosmology-major rows: for each source (sim, z0), the m evenly spaced slices."""
    m = int(multiplicity)
    if m < 1 or z_size % m:
        raise ValueError("multiplicity must divide the z size")
    step = z_size // m
    rows = []
    for sim, z0 in source_pairs:
        for j in range(m):
            rows.append((int(sim), (int(z0) + j * step) % z_size))
    pairs = np.array(rows, dtype=np.int64)
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("duplicate (sim, z) pairs in multiplicity selection")
    return pairs


def _write_pairs(path: Path, pairs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "simulation_index", "z_index"])
        for index, (simulation, z_index) in enumerate(pairs):
            writer.writerow([index, int(simulation), int(z_index)])


def manifest_row(*, multiplicity: int, config_path: Path, config: dict[str, Any], image_file: Path,
                 label_file: Path, raw_label_file: Path, pairs_file: Path, heldout_indices_path: Path,
                 heldout_norm_path: Path, heldout_raw_path: Path, norm_info: dict[str, Any],
                 source_run: str) -> dict[str, Any]:
    size = dataset_size(multiplicity)
    steps = base.steps_per_epoch(size)
    epochs = base.epochs_for(size, base.TARGET_UPDATES)
    checkpoint_epoch = epochs - 1
    name = run_name(multiplicity)
    return {
        "run_name": name, "source_run_name": None, "reuse_existing_checkpoint": False,
        "initialization": "fresh", "training_seed": TRAINING_SEED, "regime": "fixedC64_multiplicity",
        "arch": "u128", "dataset_size": int(size), "multiplicity": int(multiplicity),
        "n_cosmologies": N_COSMOLOGIES, "source_selection_run": source_run,
        "steps_per_epoch": int(steps), "epochs": int(epochs), "target_updates": int(base.TARGET_UPDATES),
        "actual_updates": int(steps * epochs), "batch_size": int(base.BATCH_SIZE),
        "conditioning": "continuous", "condition_dim": len(PARAM_NAMES), "param_names": list(PARAM_NAMES),
        "prepared_image_path": str(image_file), "train_label_path": str(label_file),
        "train_raw_params_path": str(raw_label_file), "selected_pairs_path": str(pairs_file),
        "heldout_indices_path": str(heldout_indices_path),
        "heldout_sample_params_norm_path": str(heldout_norm_path),
        "heldout_raw_params_path": str(heldout_raw_path),
        "heldout_samples_per_cosmology": int(base.SAMPLE_K_PER_COSMOLOGY),
        "normalization": {"transform": ["log"], "method": "tanh", **norm_info},
        "config": str(config_path.relative_to(config_path.parents[3])),
        "checkpoint_dir": str(config["io"]["output_dir"]), "checkpoint_epoch": int(checkpoint_epoch),
        "requested_checkpoint": str(Path(config["io"]["output_dir"]) / f"checkpoint-epoch-{checkpoint_epoch:04d}"),
        "sample_path": f"results/{SWEEP_NAME}/samples/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
        "note": ("Fixed 64 training cosmologies (identical to the N=64 full-sweep run); "
                 f"{multiplicity} evenly spaced z-slices per simulation. Held-out CAMELS 900-931."),
    }


def prepare(project_dir: Path, data_root: Path, checkpoint_root: Path, prepared_root: Path,
            write_arrays: bool) -> list[dict[str, Any]]:
    norm_info, param_stats, heldout, heldout_indices_path, heldout_norm_path, heldout_raw_path = \
        full._load_shared_state(project_dir)
    raw_params = load_params(params_path(data_root), 1000)
    mean = np.asarray(param_stats["mean"], dtype=np.float32)
    std = np.asarray(param_stats["std"], dtype=np.float32)
    normalized_params = ((raw_params - mean) / std).astype(np.float32)
    grid_path = image_path(data_root)

    source_pairs = load_source_pairs(project_dir)
    if np.intersect1d(source_pairs[:, 0], heldout).size:
        raise ValueError("source training simulations overlap the held-out set")
    source_run = full.run_name(N_COSMOLOGIES)

    local_root = project_dir / "local" / SWEEP_NAME
    config_dir, label_dir = local_root / "configs", local_root / "labels"
    config_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for m in MULTIPLICITIES:
        name = run_name(m)
        pairs = multiplicity_pairs(source_pairs, m)
        if m == 1 and not np.array_equal(pairs, source_pairs):
            raise AssertionError("m=1 selection must equal the existing N=64 selection exactly")
        config_path = config_dir / f"{name}.yaml"
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
            checkpoint_root=checkpoint_root, prepared_data_root=prepared_root,
            dataset_size=dataset_size(m), norm_info=norm_info, image_file=image_file,
            label_file=label_file, heldout_label_file=heldout_norm_path,
            heldout_count=len(heldout), sample_k=base.SAMPLE_K_PER_COSMOLOGY)
        config["io"]["output_dir"] = str(checkpoint_root / f"{name}_checkpoints")
        config["generate"]["continuous_labels"] = str(heldout_norm_path)
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        # Protocol must match the source sweep exactly except for the data files.
        source_config_path = project_dir / "local" / SOURCE_SWEEP_NAME / "configs" / f"{source_run}.yaml"
        if source_config_path.is_file():
            source_config = yaml.safe_load(source_config_path.read_text())
            full.assert_matching_scientific_protocol(
                {**source_config, "train": {**source_config["train"], "num_epochs": config["train"]["num_epochs"]}},
                config)
        rows.append(manifest_row(
            multiplicity=m, config_path=config_path, config=config, image_file=image_file,
            label_file=label_file, raw_label_file=raw_label_file, pairs_file=pairs_file,
            heldout_indices_path=heldout_indices_path, heldout_norm_path=heldout_norm_path,
            heldout_raw_path=heldout_raw_path, norm_info=norm_info, source_run=source_run))
        print(f"Wrote {config_path}  (m={m}, N={dataset_size(m)}, epochs={rows[-1]['epochs']}, "
              f"checkpoint_epoch={rows[-1]['checkpoint_epoch']})")
    (local_root / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {local_root / 'manifest.json'}")
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
    if args.print_runs or args.print_train_runs:
        for m in MULTIPLICITIES:
            print(run_name(m))
        return
    rows = prepare(Path(args.project_dir).resolve(), Path(args.data_root), Path(args.checkpoint_root),
                   Path(args.prepared_data_root), args.write_arrays)
    print("Prepared:", ", ".join(f"m={r['multiplicity']} N={r['dataset_size']}" for r in rows))
    print("Training RNG seed:", TRAINING_SEED)


if __name__ == "__main__":
    main()
