#!/usr/bin/env python
"""Prepare the label-variant conditional sweeps requested by Nick.

All reuse the full sweep's exact nested training selections for each N (same
(sim, z) rows, same order), the same held-out split, image normalization, model
width, scheduler and 200k-update budget. Only the conditioning label changes:

  omsig_continuous : continuous conditioning on (Omega_m, sigma_8) only
                     (UNet2DConditionModel, encoder_hid_dim=2; labels = the first
                     two columns of the frozen six-parameter normalization).
  om_continuous    : continuous conditioning on Omega_m alone (encoder_hid_dim=1;
                     labels = the first column, kept 2-D as (N, 1)). Reuses the
                     omsig_continuous prepared training images (same selection).
  omsig_class36    : class conditioning on 6 x 6 uniform bins in Omega_m in
                     [0.1, 0.5] and sigma_8 in [0.6, 1.0]; class = 6*i_Om + i_s8
                     (UNet2DModel with class embeddings; num_class_embeds = 37,
                     the last one an unused null class; cfg_dropout = 0).

Held-out sampling requests the 32 held-out cosmologies (64 draws each) either as
their 2-D normalized parameters or as their class id; evaluation still compares
the frozen probe's recovery with the true six-parameter values (theta_raw).

Default sizes are the five-point subset 64, 256, 1024, 4096, 32768; pass
--sizes 64,128,... for more. Writes only under local/<sweep>/ and the prepared
data root; no training or sampling here.
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

VARIANTS = {
    "omsig_continuous": "nf_conditional_omsig_continuous_200k",
    "omsig_class36": "nf_conditional_omsig_class36_200k",
    "om_continuous": "nf_conditional_om_continuous_200k",
}
CONTINUOUS_COLUMNS = {"omsig_continuous": 2, "om_continuous": 1}  # leading columns of the six-parameter vector
DEFAULT_SIZES = (64, 256, 1024, 4096, 32768)
TRAINING_SEED = full.TRAINING_SEED
OMEGA_EDGES = np.linspace(0.1, 0.5, 7)
SIGMA_EDGES = np.linspace(0.6, 1.0, 7)
N_BINS = 6
N_CLASSES = N_BINS * N_BINS


def sweep_name(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    return VARIANTS[variant]


def run_name(variant: str, dataset_size: int) -> str:
    exponent = int(round(math.log2(int(dataset_size))))
    if 2**exponent != int(dataset_size):
        raise ValueError(f"dataset_size must be a power of two, got {dataset_size}")
    tag = {"omsig_continuous": "omsigc", "omsig_class36": "omsig36", "om_continuous": "omc"}[variant]
    return f"nf_cond_{tag}_hi_u128_d2p{exponent:02d}_n{int(dataset_size)}_fresh200k"


def class_ids(raw_params: np.ndarray) -> np.ndarray:
    """6x6 uniform bins over the CAMELS LH ranges; edges are inclusive at the top."""
    raw = np.asarray(raw_params, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("raw parameters must have at least Omega_m and sigma_8 columns")
    i_om = np.clip(np.searchsorted(OMEGA_EDGES, raw[:, 0], side="right") - 1, 0, N_BINS - 1)
    i_s8 = np.clip(np.searchsorted(SIGMA_EDGES, raw[:, 1], side="right") - 1, 0, N_BINS - 1)
    if np.any(raw[:, 0] < OMEGA_EDGES[0] - 1e-6) or np.any(raw[:, 0] > OMEGA_EDGES[-1] + 1e-6):
        raise ValueError("Omega_m outside the CAMELS LH range")
    if np.any(raw[:, 1] < SIGMA_EDGES[0] - 1e-6) or np.any(raw[:, 1] > SIGMA_EDGES[-1] + 1e-6):
        raise ValueError("sigma_8 outside the CAMELS LH range")
    return (N_BINS * i_om + i_s8).astype(np.int64)


def class_centers() -> np.ndarray:
    om = 0.5 * (OMEGA_EDGES[:-1] + OMEGA_EDGES[1:])
    s8 = 0.5 * (SIGMA_EDGES[:-1] + SIGMA_EDGES[1:])
    return np.array([[om[i], s8[j]] for i in range(N_BINS) for j in range(N_BINS)])


def variant_config(variant: str, config: dict[str, Any], label_file: Path, heldout_label_file: Path,
                   output_dir: Path) -> dict[str, Any]:
    """Apply the label-variant changes to a full-sweep config."""
    config = json.loads(json.dumps(config))
    config["io"]["output_dir"] = str(output_dir)
    config["data"]["label_path"] = str(label_file)
    if variant in CONTINUOUS_COLUMNS:
        config["model"]["kwargs"]["encoder_hid_dim"] = CONTINUOUS_COLUMNS[variant]
        config["train"]["conditioning"] = "continuous"
        config["generate"]["conditioning"] = "continuous"
        config["generate"]["continuous_labels"] = str(heldout_label_file)
        config["generate"]["labels"] = None
    else:
        config["model"] = {
            "class": "UNet2DModel",
            "kwargs": {
                "sample_size": 128, "in_channels": 1, "out_channels": 1, "layers_per_block": 2,
                "block_out_channels": [32, 64, 128],
                "down_block_types": ["DownBlock2D", "DownBlock2D", "AttnDownBlock2D"],
                "up_block_types": ["AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
                "norm_num_groups": 32,
                "num_class_embeds": N_CLASSES + 1,   # reserve one unused null class
            },
        }
        config["train"]["conditioning"] = "discrete"
        config["train"]["cfg_dropout"] = 0.0
        config["generate"]["conditioning"] = "discrete"
        config["generate"]["labels"] = str(heldout_label_file)
        config["generate"]["continuous_labels"] = None
    return config


def manifest_row(*, variant: str, dataset_size: int, config_path: Path, config: dict[str, Any],
                 image_file: Path, label_file: Path, raw_label_file: Path, pairs_file: Path,
                 heldout_indices_path: Path, heldout_label_file: Path, heldout_raw_path: Path,
                 norm_info: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    steps = base.steps_per_epoch(dataset_size)
    epochs = base.epochs_for(dataset_size, base.TARGET_UPDATES)
    checkpoint_epoch = epochs - 1
    name = run_name(variant, dataset_size)
    sweep = sweep_name(variant)
    row = {
        "run_name": name, "source_run_name": None, "reuse_existing_checkpoint": False,
        "initialization": "fresh", "training_seed": TRAINING_SEED, "regime": variant,
        "arch": "u128", "dataset_size": int(dataset_size), "steps_per_epoch": int(steps),
        "epochs": int(epochs), "target_updates": int(base.TARGET_UPDATES),
        "actual_updates": int(steps * epochs), "batch_size": int(base.BATCH_SIZE),
        "conditioning": config["train"]["conditioning"], "param_names": list(PARAM_NAMES),
        "prepared_image_path": str(image_file), "train_label_path": str(label_file),
        "train_raw_params_path": str(raw_label_file), "selected_pairs_path": str(pairs_file),
        "heldout_indices_path": str(heldout_indices_path),
        "heldout_sample_params_norm_path": str(heldout_label_file),
        "heldout_raw_params_path": str(heldout_raw_path),
        "heldout_samples_per_cosmology": int(base.SAMPLE_K_PER_COSMOLOGY),
        "normalization": {"transform": ["log"], "method": "tanh", **norm_info},
        "config": str(config_path.relative_to(config_path.parents[3])),
        "checkpoint_dir": str(config["io"]["output_dir"]), "checkpoint_epoch": int(checkpoint_epoch),
        "requested_checkpoint": str(Path(config["io"]["output_dir"]) / f"checkpoint-epoch-{checkpoint_epoch:04d}"),
        "sample_path": f"results/{sweep}/samples/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
        "training_selection": f"identical to {full.SWEEP_NAME} at this N",
        "note": f"Label variant {variant}; held-out CAMELS simulations 900-931.",
    }
    row.update(extra)
    return row


def prepare(variant: str, project_dir: Path, data_root: Path, sizes: tuple[int, ...],
            write_arrays: bool) -> list[dict[str, Any]]:
    sweep = sweep_name(variant)
    checkpoint_root = Path(f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{sweep}")
    prepared_root = checkpoint_root / "prepared_data"
    norm_info, param_stats, heldout, heldout_indices_path, heldout_norm_path, heldout_raw_path = \
        full._load_shared_state(project_dir)
    raw_params = load_params(params_path(data_root), 1000)
    mean = np.asarray(param_stats["mean"], dtype=np.float32)
    std = np.asarray(param_stats["std"], dtype=np.float32)
    normalized_params = ((raw_params - mean) / std).astype(np.float32)
    grid_path = image_path(data_root)
    allowed = base.allowed_indices(len(raw_params), heldout)
    selected = full.nested_slice_pairs_by_size(allowed, full.ALL_DATASET_SIZES, z_size=128)

    local_root = project_dir / "local" / sweep
    config_dir, label_dir, heldout_dir = local_root / "configs", local_root / "labels", local_root / "heldout"
    for d in (config_dir, label_dir, heldout_dir):
        d.mkdir(parents=True, exist_ok=True)

    # held-out request labels, cosmology-major with k repeats (same layout as the full sweep)
    k = base.SAMPLE_K_PER_COSMOLOGY
    heldout_raw = np.load(heldout_raw_path, allow_pickle=False)
    if heldout_raw.shape != (len(heldout), 6):
        raise ValueError("held-out raw parameters have an unexpected shape")
    extra: dict[str, Any]
    if variant in CONTINUOUS_COLUMNS:
        ncol = CONTINUOUS_COLUMNS[variant]
        heldout_label = np.repeat(((heldout_raw - mean) / std)[:, :ncol].astype(np.float32), k, axis=0)
        heldout_label_file = heldout_dir / f"heldout_params_norm_{'omsig' if ncol == 2 else 'om'}_k{k}.npy"
        extra = {"condition_dim": ncol, "conditioned_parameters": list(PARAM_NAMES[:ncol])}
    else:
        heldout_label = np.repeat(class_ids(heldout_raw), k)
        heldout_label_file = heldout_dir / f"heldout_class_labels_k{k}.npy"
        extra = {"condition_dim": 1, "n_classes": N_CLASSES, "num_class_embeds": N_CLASSES + 1,
                 "omega_bin_edges": OMEGA_EDGES.tolist(), "sigma8_bin_edges": SIGMA_EDGES.tolist(),
                 "class_centers_path": str(heldout_dir / "class_centers_omsig.npy"),
                 "heldout_class_ids_path": str(heldout_dir / "heldout_class_ids.npy")}
        np.save(heldout_dir / "class_centers_omsig.npy", class_centers().astype(np.float32))
        np.save(heldout_dir / "heldout_class_ids.npy", class_ids(heldout_raw))
    np.save(heldout_label_file, heldout_label)

    source_config_dir = project_dir / "local" / full.SWEEP_NAME / "configs"
    rows: list[dict[str, Any]] = []
    for size in sizes:
        name = run_name(variant, size)
        pairs = selected[size]
        config_path = config_dir / f"{name}.yaml"
        pairs_file = label_dir / f"{name}_selected_slices.csv"
        image_file = prepared_root / f"{name}_train_images.npy"
        label_file = label_dir / f"{name}_train_labels.npy"
        raw_label_file = label_dir / f"{name}_train_params_raw.npy"
        reuse_images = variant == "om_continuous"
        if reuse_images:  # identical selection; reuse the omsig_continuous prepared maps
            omsig_name = run_name("omsig_continuous", size)
            image_file = (checkpoint_root.parent / VARIANTS["omsig_continuous"] / "prepared_data"
                          / f"{omsig_name}_train_images.npy")
            omsig_pairs_file = (project_dir / "local" / VARIANTS["omsig_continuous"] / "labels"
                                / f"{omsig_name}_selected_slices.csv")
            with omsig_pairs_file.open() as handle:
                omsig_pairs = np.array([[int(r["simulation_index"]), int(r["z_index"])] for r in csv.DictReader(handle)])
            if not np.array_equal(omsig_pairs, pairs):
                raise AssertionError(f"N={size}: selection differs from omsig_continuous; cannot reuse its images")
            if np.load(image_file, mmap_mode="r").shape[0] != len(pairs):
                raise AssertionError(f"N={size}: {image_file} has the wrong number of maps")
        full._write_pairs(pairs_file, pairs)
        sims = pairs[:, 0]
        if variant in CONTINUOUS_COLUMNS:
            labels = normalized_params[sims][:, :CONTINUOUS_COLUMNS[variant]].astype(np.float32)
        else:
            labels = class_ids(raw_params[sims])
        if write_arrays:
            if not reuse_images:
                base.materialize_slices(grid_path=grid_path, out_path=image_file, pairs=pairs)
            np.save(label_file, labels)
            np.save(raw_label_file, raw_params[sims])
        # verify the selection is exactly the full sweep's for this N, when that sweep is present
        source_pairs_file = (project_dir / "local" / full.SWEEP_NAME / "labels"
                             / f"{full.run_name(size)}_selected_slices.csv")
        if source_pairs_file.is_file():
            with source_pairs_file.open() as handle:
                src = np.array([[int(r["simulation_index"]), int(r["z_index"])] for r in csv.DictReader(handle)])
            if not np.array_equal(src, pairs):
                raise AssertionError(f"N={size}: selection differs from the full sweep's")
        base_config = base.build_config(
            checkpoint_root=checkpoint_root, prepared_data_root=prepared_root, dataset_size=size,
            norm_info=norm_info, image_file=image_file, label_file=label_file,
            heldout_label_file=heldout_label_file, heldout_count=len(heldout), sample_k=k)
        config = variant_config(variant, base_config, label_file, heldout_label_file,
                                checkpoint_root / f"{name}_checkpoints")
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        row = manifest_row(variant=variant, dataset_size=size, config_path=config_path, config=config,
                           image_file=image_file, label_file=label_file, raw_label_file=raw_label_file,
                           pairs_file=pairs_file, heldout_indices_path=heldout_indices_path,
                           heldout_label_file=heldout_label_file, heldout_raw_path=heldout_raw_path,
                           norm_info=norm_info, extra=extra)
        if variant == "omsig_class36":
            counts = np.bincount(labels, minlength=N_CLASSES)
            row["train_maps_per_class_mean"] = float(counts.mean())
            row["train_classes_populated"] = int(np.sum(counts > 0))
        rows.append(row)
        print(f"Wrote {config_path}  (N={size}, epochs={row['epochs']}, checkpoint_epoch={row['checkpoint_epoch']}"
              + (f", maps/class={row['train_maps_per_class_mean']:.2f}, classes populated={row['train_classes_populated']}/36"
                 if variant == "omsig_class36" else "") + ")")
    (local_root / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {local_root / 'manifest.json'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--data-root", default=base.DATA_ROOT)
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--write-arrays", action="store_true")
    parser.add_argument("--print-runs", action="store_true")
    parser.add_argument("--print-train-runs", action="store_true")
    args = parser.parse_args()
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())
    bad = sorted(set(sizes) - set(full.ALL_DATASET_SIZES))
    if bad:
        raise SystemExit(f"unsupported sizes {bad}")
    if args.print_runs or args.print_train_runs:
        for size in sizes:
            print(run_name(args.variant, size))
        return
    rows = prepare(args.variant, Path(args.project_dir).resolve(), Path(args.data_root), sizes, args.write_arrays)
    print("Prepared:", ", ".join(f"N={r['dataset_size']}" for r in rows), "| variant", args.variant)


if __name__ == "__main__":
    main()
