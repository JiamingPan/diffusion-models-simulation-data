#!/usr/bin/env python
"""Prepare CFG/guidance ablation configs for the continuous HI bias probe.

This script deliberately writes a separate sweep namespace from the v1
``nf_conditional_bias_probe`` jobs so the running CFG-off chain is not touched.
It reuses the exact materialized training arrays, labels, held-out cosmologies,
and image normalization produced by ``prepare_nf_conditional_bias_probe_configs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

import prepare_nf_conditional_bias_probe_configs as base


SWEEP_NAME = "nf_conditional_bias_probe_cfg_sweep"
BASE_SWEEP_NAME = base.SWEEP_NAME
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
DEFAULT_CFG_DROPOUTS = (0.05, 0.1)
DEFAULT_GUIDANCE_SCALES = (None, 1.0, 1.5, 2.0)


def float_slug(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def guidance_label(value: float | None) -> str:
    if value is None:
        return "noguidance"
    return f"g{float_slug(float(value))}"


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_guidance_list(value: str) -> list[float | None]:
    out: list[float | None] = []
    for item in value.split(","):
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        if cleaned in {"none", "null", "noguidance", "no_guidance"}:
            out.append(None)
        else:
            out.append(float(cleaned))
    return out


def cfg_run_name(dataset_size: int, cfg_dropout: float) -> str:
    return f"{base.run_name(dataset_size)}_cfgdrop{float_slug(cfg_dropout)}"


def base_paths(project_dir: Path, dataset_size: int, sample_k: int) -> dict[str, Path]:
    base_name = base.run_name(dataset_size)
    root = project_dir / "local" / BASE_SWEEP_NAME
    return {
        "base_manifest": root / "manifest.json",
        "base_config": root / "configs" / f"{base_name}.yaml",
        "base_image": Path(base.PREPARED_DATA_ROOT) / f"{base_name}_train_images.npy",
        "base_labels": root / "labels" / f"{base_name}_train_params_norm.npy",
        "base_raw_labels": root / "labels" / f"{base_name}_train_params_raw.npy",
        "base_pairs": root / "labels" / f"{base_name}_selected_slices.csv",
        "heldout_indices": root / "heldout" / "heldout_simulation_indices.txt",
        "heldout_raw": root / "heldout" / "heldout_params_raw.npy",
        "heldout_repeated": root / "heldout" / f"heldout_params_norm_k{sample_k}.npy",
        "norm_info": root / "heldout" / "shared_image_norm_stats.json",
    }


def require_paths(paths: dict[str, Path]) -> None:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        detail = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Missing base bias-probe prepared files. Run the v1 prepare job first:\n"
            "  sbatch -A huterer2 scripts/slurm/prepare_nf_conditional_bias_probe.sbatch\n"
            f"Missing:\n{detail}"
        )


def build_cfg_config(
    *,
    project_dir: Path,
    checkpoint_root: Path,
    dataset_size: int,
    cfg_dropout: float,
    sample_k: int,
) -> dict[str, Any]:
    paths = base_paths(project_dir, dataset_size, sample_k)
    require_paths(paths)
    norm_info = json.loads(paths["norm_info"].read_text())
    heldout_count = len((project_dir / "local" / BASE_SWEEP_NAME / "heldout" / "heldout_simulation_indices.txt").read_text().splitlines())
    cfg = base.build_config(
        checkpoint_root=checkpoint_root,
        prepared_data_root=Path(base.PREPARED_DATA_ROOT),
        dataset_size=dataset_size,
        norm_info=norm_info,
        image_file=paths["base_image"],
        label_file=paths["base_labels"],
        heldout_label_file=paths["heldout_repeated"],
        heldout_count=heldout_count,
        sample_k=sample_k,
    )
    name = cfg_run_name(dataset_size, cfg_dropout)
    cfg["io"]["output_dir"] = str(checkpoint_root / f"{name}_checkpoints")
    cfg["train"]["cfg_dropout"] = float(cfg_dropout)
    cfg["generate"]["guidance_scale"] = None
    return cfg


def manifest_row(
    *,
    project_dir: Path,
    checkpoint_root: Path,
    dataset_size: int,
    cfg_dropout: float,
    guidance_scales: list[float | None],
    sample_k: int,
) -> dict[str, Any]:
    paths = base_paths(project_dir, dataset_size, sample_k)
    require_paths(paths)
    name = cfg_run_name(dataset_size, cfg_dropout)
    spe = base.steps_per_epoch(dataset_size)
    epochs = base.epochs_for(dataset_size, base.TARGET_UPDATES)
    return {
        "run_name": name,
        "base_run_name": base.run_name(dataset_size),
        "regime": "memorization" if int(dataset_size) <= 128 else "generalization",
        "arch": "u128",
        "field": base.FIELD,
        "simulation": base.SIM,
        "dataset": base.DATASET,
        "redshift": float(base.REDSHIFT),
        "resolution": base.RESOLUTION,
        "dataset_size": int(dataset_size),
        "steps_per_epoch": int(spe),
        "epochs": int(epochs),
        "target_updates": int(base.TARGET_UPDATES),
        "actual_updates": int(spe * epochs),
        "checkpoint_every_updates": int(base.CHECKPOINT_EVERY_UPDATES),
        "checkpoint_every_n_epochs": int(base.checkpoint_epochs_for(dataset_size)),
        "batch_size": int(base.BATCH_SIZE),
        "conditioning": "continuous",
        "cfg_dropout": float(cfg_dropout),
        "guidance_scales": [None if x is None else float(x) for x in guidance_scales],
        "condition_dim": len(base.PARAM_NAMES),
        "param_names": base.PARAM_NAMES,
        "prepared_image_path": str(paths["base_image"]),
        "train_label_path": str(paths["base_labels"]),
        "train_raw_params_path": str(paths["base_raw_labels"]),
        "selected_pairs_path": str(paths["base_pairs"]),
        "heldout_indices_path": str(paths["heldout_indices"]),
        "heldout_raw_params_path": str(paths["heldout_raw"]),
        "heldout_sample_params_norm_path": str(paths["heldout_repeated"]),
        "heldout_samples_per_cosmology": int(sample_k),
        "config": f"local/{SWEEP_NAME}/configs/{name}.yaml",
        "checkpoint_dir": str(checkpoint_root / f"{name}_checkpoints"),
        "sample_path": (
            f"results/{SWEEP_NAME}/samples/"
            f"{name}_seed{{seed}}_dpm50_heldout_k{{k}}_{{guidance}}.npz"
        ),
        "note": (
            "CFG/guidance ablation for the continuous HI bias probe. "
            "Uses the same prepared real data and held-out cosmologies as the CFG-off v1 run."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--dataset-sizes", default=",".join(str(x) for x in base.DATASET_SIZES))
    parser.add_argument("--cfg-dropouts", default=",".join(str(x) for x in DEFAULT_CFG_DROPOUTS))
    parser.add_argument(
        "--guidance-scales",
        default=",".join("none" if x is None else str(x) for x in DEFAULT_GUIDANCE_SCALES),
    )
    parser.add_argument("--sample-k-per-cosmology", type=int, default=base.SAMPLE_K_PER_COSMOLOGY)
    parser.add_argument("--print-train-runs", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    checkpoint_root = Path(args.checkpoint_root)
    dataset_sizes = base.parse_int_list(args.dataset_sizes)
    cfg_dropouts = parse_float_list(args.cfg_dropouts)
    guidance_scales = parse_guidance_list(args.guidance_scales)

    if args.print_train_runs:
        for dataset_size in dataset_sizes:
            for cfg_dropout in cfg_dropouts:
                print(cfg_run_name(dataset_size, cfg_dropout))
        return

    config_dir = project_dir / "local" / SWEEP_NAME / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset_size in dataset_sizes:
        for cfg_dropout in cfg_dropouts:
            name = cfg_run_name(dataset_size, cfg_dropout)
            cfg = build_cfg_config(
                project_dir=project_dir,
                checkpoint_root=checkpoint_root,
                dataset_size=dataset_size,
                cfg_dropout=cfg_dropout,
                sample_k=args.sample_k_per_cosmology,
            )
            config_path = config_dir / f"{name}.yaml"
            with config_path.open("w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            rows.append(
                manifest_row(
                    project_dir=project_dir,
                    checkpoint_root=checkpoint_root,
                    dataset_size=dataset_size,
                    cfg_dropout=cfg_dropout,
                    guidance_scales=guidance_scales,
                    sample_k=args.sample_k_per_cosmology,
                )
            )
            print(f"Wrote {config_path}")

    manifest_path = project_dir / "local" / SWEEP_NAME / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {manifest_path}")

    if args.print_table:
        cols = [
            "run_name",
            "regime",
            "dataset_size",
            "cfg_dropout",
            "epochs",
            "actual_updates",
            "checkpoint_every_n_epochs",
        ]
        print("\t".join(cols))
        for row in rows:
            print("\t".join(str(row[col]) for col in cols))
        print("guidance scales:", ",".join(guidance_label(x) for x in guidance_scales))


if __name__ == "__main__":
    main()
