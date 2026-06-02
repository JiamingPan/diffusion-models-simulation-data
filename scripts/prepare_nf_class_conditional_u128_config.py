#!/usr/bin/env python
"""Prepare a u128 discrete class-conditional CAMELS field-type run.

This is the simpler conditional baseline requested by Nick:

- condition on a discrete field label only, not cosmological parameters
- use labels such as 0=Mcdm, 1=Mstar, 2=HI, ...
- train one u128 UNet2DModel with class embeddings

The generated label file cycles through the requested field classes, so the
sample output can be split by class after generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from prepare_nf_generalize_nick_data_configs import EMA_SIGMA_RELS, ZTHIN


SWEEP_NAME = "nf_class_conditional_u128"
DATA_ROOT = "/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG"
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
SIM = "IllustrisTNG"
DATASET = "LH"
RESOLUTION = 128
REDSHIFT = "0.0"
FIELDS = ["Mcdm", "Mstar", "HI", "Mgas", "Mtot", "ne"]
N_TRAIN_SIMS = 1000
TARGET_UPDATES = 200_000
CHECKPOINT_EVERY_UPDATES = 20_000
SAMPLE_N = 512
BATCH_SIZE = 32
RUN_VARIANT = "logfloor"
PER_FIELD_RUN_VARIANT = "logfloor_perfieldnorm"
NORMALIZATION_MODES = {"shared", "per-field", "per_field"}


class ManifestLock:
    def __init__(self, path: Path, timeout_s: int = 120):
        self.path = path
        self.timeout_s = timeout_s

    def __enter__(self) -> None:
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self.path.mkdir(parents=True)
                return None
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for manifest lock: {self.path}")
                time.sleep(0.25)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.path.rmdir()
        except FileNotFoundError:
            pass


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else int(default)


def parse_fields(value: str) -> list[str]:
    fields = [x.strip() for x in value.split(",") if x.strip()]
    if not fields:
        raise ValueError("Need at least one field.")
    return fields


def dataset_slug(fields: list[str], n_train_sims: int) -> str:
    field_slug = "-".join(field.lower() for field in fields)
    return f"{field_slug}_lh_z0p0_n{n_train_sims}"


def normalize_mode(value: str | None) -> str:
    mode = (value or os.environ.get("NF_CLASS_NORMALIZATION_MODE", "shared")).strip().lower()
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode {value!r}; use shared or per-field.")
    return "per-field" if mode == "per_field" else mode


def run_variant(normalization_mode: str | None = None) -> str:
    return PER_FIELD_RUN_VARIANT if normalize_mode(normalization_mode) == "per-field" else RUN_VARIANT


def updates_slug(target_updates: int) -> str:
    target_updates = int(target_updates)
    if target_updates % 1_000_000 == 0:
        return f"{target_updates // 1_000_000}m"
    if target_updates % 1000 == 0:
        return f"{target_updates // 1000}k"
    return str(target_updates)


def run_name(
    fields: list[str] | None = None,
    n_train_sims: int = N_TRAIN_SIMS,
    normalization_mode: str | None = None,
    target_updates: int = TARGET_UPDATES,
) -> str:
    fields = fields or FIELDS
    return (
        f"nf_class_u128_{dataset_slug(fields, n_train_sims)}_"
        f"{run_variant(normalization_mode)}_{updates_slug(target_updates)}"
    )


def image_path(data_root: str | Path, field: str) -> Path:
    return Path(data_root) / f"Grids_{field}_{SIM}_{DATASET}_{RESOLUTION}_z={REDSHIFT}.npy"


def slices_per_sim() -> int:
    return 128 // ZTHIN


def dataset_size(n_train_sims: int, n_fields: int) -> int:
    return int(n_train_sims) * slices_per_sim() * int(n_fields)


def steps_per_epoch(n_train_sims: int, n_fields: int, batch_size: int = BATCH_SIZE) -> int:
    return max(1, math.ceil(dataset_size(n_train_sims, n_fields) / int(batch_size)))


def epochs_for(n_train_sims: int, n_fields: int, target_updates: int = TARGET_UPDATES) -> int:
    return max(1, math.ceil(int(target_updates) / steps_per_epoch(n_train_sims, n_fields)))


def checkpoint_epochs_for(
    n_train_sims: int,
    n_fields: int,
    checkpoint_every_updates: int = CHECKPOINT_EVERY_UPDATES,
) -> int:
    return max(1, round(int(checkpoint_every_updates) / steps_per_epoch(n_train_sims, n_fields)))


def write_labels(
    *,
    project_dir: Path,
    fields: list[str],
    n_train_sims: int,
    sample_n: int,
    normalization_mode: str,
    target_updates: int,
) -> dict[str, Any]:
    name = run_name(fields, n_train_sims, normalization_mode, target_updates)
    label_dir = project_dir / "local" / SWEEP_NAME / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    train_label_paths: list[Path] = []
    class_map = {field: i for i, field in enumerate(fields)}
    for field, class_id in class_map.items():
        labels = np.full(n_train_sims, class_id, dtype=np.int64)
        path = label_dir / f"{name}_{field}_train_class.npy"
        np.save(path, labels)
        train_label_paths.append(path)

    sample_labels = (np.arange(sample_n, dtype=np.int64) % len(fields)).astype(np.int64)
    sample_label_path = label_dir / f"{name}_sample_class_labels_n{sample_n}.npy"
    sample_counts_path = label_dir / f"{name}_sample_class_counts_n{sample_n}.json"
    class_map_path = label_dir / f"{name}_class_map.json"
    np.save(sample_label_path, sample_labels)
    sample_counts = {
        field: int(np.sum(sample_labels == class_id))
        for field, class_id in class_map.items()
    }
    sample_counts_path.write_text(json.dumps(sample_counts, indent=2) + "\n")
    class_map_path.write_text(json.dumps(class_map, indent=2) + "\n")

    return {
        "class_map": class_map,
        "train_label_paths": train_label_paths,
        "sample_label_path": sample_label_path,
        "sample_counts_path": sample_counts_path,
        "class_map_path": class_map_path,
    }


def build_config(
    *,
    data_root: Path,
    checkpoint_root: Path,
    fields: list[str],
    n_train_sims: int,
    sample_n: int,
    label_paths: dict[str, Any],
    normalization_mode: str,
    target_updates: int,
    checkpoint_every_updates: int,
) -> dict[str, Any]:
    name = run_name(fields, n_train_sims, normalization_mode, target_updates)
    n_classes = len(fields)
    base_norm_kwargs = {
        "center": None,
        "xmax": None,
        "alpha": 0.8,
        "beta": 10.0,
        "delta": 1.0,
        "gamma": 1.0,
        "sigma": 1.5,
    }
    if normalize_mode(normalization_mode) == "per-field":
        # cosmodiff MultiNormalization fits one normalization object per
        # img_path/class. This keeps sparse Mstar from sharing Mcdm/HI/Mgas
        # scaling statistics.
        normalization: str | list[str] = ["tanh"] * n_classes
        norm_kwargs: dict[str, Any] | list[dict[str, Any]] = [dict(base_norm_kwargs) for _ in fields]
        transform: list[str] | list[list[str]] = [["log"] for _ in fields]
    else:
        normalization = "tanh"
        norm_kwargs = base_norm_kwargs
        transform = ["log"]
    return {
        "global": {
            "device": "cuda",
            "dtype": "float32",
        },
        "io": {
            "output_dir": str(checkpoint_root / f"{name}_checkpoints"),
        },
        "data": {
            "img_path": [str(image_path(data_root, field)) for field in fields],
            "img_read_fn": "npy_read_fn",
            "label_path": [str(path) for path in label_paths["train_label_paths"]],
            "label_read_fn": "npy_read_fn",
            "reshape": "2d",
            "zthin": ZTHIN,
            "n_samples": [int(n_train_sims)] * n_classes,
            "seed": None,
            "keep_on_cpu": True,
            "normalization": normalization,
            "norm_kwargs": norm_kwargs,
            "transform": transform,
        },
        "model": {
            "class": "UNet2DModel",
            "kwargs": {
                "sample_size": 128,
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 2,
                "block_out_channels": [32, 64, 128],
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
                "norm_num_groups": 32,
                # Reserve one unused null class so CFG can be tried later
                # without changing checkpoint shape.
                "num_class_embeds": n_classes + 1,
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
            "num_epochs": int(epochs_for(n_train_sims, n_classes, target_updates)),
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "checkpoint_every_n_epochs": int(
                checkpoint_epochs_for(n_train_sims, n_classes, checkpoint_every_updates)
            ),
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
            "n_samples": int(sample_n),
            "batch_size": None,
            "image_shape": None,
            "conditioning": "discrete",
            "labels": str(label_paths["sample_label_path"]),
            "continuous_labels": None,
            "guidance_scale": None,
            "ema_sigma_rel": None,
            "seed": None,
            "device": None,
        },
    }


def manifest_row(
    *,
    data_root: Path,
    checkpoint_root: Path,
    fields: list[str],
    n_train_sims: int,
    sample_n: int,
    label_paths: dict[str, Any],
    normalization_mode: str,
    target_updates: int,
    checkpoint_every_updates: int,
) -> dict[str, Any]:
    mode = normalize_mode(normalization_mode)
    name = run_name(fields, n_train_sims, mode, target_updates)
    n_classes = len(fields)
    spe = steps_per_epoch(n_train_sims, n_classes)
    epochs = epochs_for(n_train_sims, n_classes, target_updates)
    return {
        "run_name": name,
        "arch": "u128",
        "arch_label": "UNet-128 class conditional",
        "variant_tag": "discrete_field_class",
        "variant_label": "u128 conditional on field type, log floor",
        "fields": fields,
        "class_map": label_paths["class_map"],
        "simulation": SIM,
        "dataset": DATASET,
        "redshift": float(REDSHIFT),
        "resolution": RESOLUTION,
        "n_train_simulations_per_field": int(n_train_sims),
        "zthin": ZTHIN,
        "slices_per_sim": slices_per_sim(),
        "dataset_size": dataset_size(n_train_sims, n_classes),
        "steps_per_epoch": int(spe),
        "epochs": int(epochs),
        "target_updates": int(target_updates),
        "actual_updates": int(spe * epochs),
        "checkpoint_every_updates": int(checkpoint_every_updates),
        "checkpoint_every_n_epochs": int(
            checkpoint_epochs_for(n_train_sims, n_classes, checkpoint_every_updates)
        ),
        "batch_size": BATCH_SIZE,
        "conditioning": "discrete",
        "normalization_mode": mode,
        "normalization_scope": "per field class" if mode == "per-field" else "shared across all field classes",
        "condition_dim": 1,
        "num_classes": n_classes,
        "num_class_embeds": n_classes + 1,
        "sample_n": int(sample_n),
        "data_paths": [str(image_path(data_root, field)) for field in fields],
        "train_label_paths": [str(path) for path in label_paths["train_label_paths"]],
        "sample_label_path": str(label_paths["sample_label_path"]),
        "sample_counts_path": str(label_paths["sample_counts_path"]),
        "class_map_path": str(label_paths["class_map_path"]),
        "config": f"local/{SWEEP_NAME}/configs/{name}.yaml",
        "checkpoint_dir": str(checkpoint_root / f"{name}_checkpoints"),
        "sample_path": f"results/{SWEEP_NAME}/samples/{name}_seed{{seed}}_raw_class_conditional.npz",
        "note": (
            "Discrete field-class conditional run requested by Nick; no cosmology-parameter conditioning. "
            "Uses a safe log floor in cosmo_diffusion to avoid non-finite zero-valued field voxels. "
            f"Normalization mode: {mode}."
        ),
    }


def assert_config(config_path: Path, row: dict[str, Any]) -> None:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    labels = [np.load(path) for path in row["train_label_paths"]]
    failed: list[str] = []
    checks = {
        "model.class": config["model"]["class"] == "UNet2DModel",
        "model.num_class_embeds": config["model"]["kwargs"].get("num_class_embeds") == row["num_class_embeds"],
        "train.conditioning": config["train"].get("conditioning") == "discrete",
        "generate.conditioning": config["generate"].get("conditioning") == "discrete",
        "generate.labels": config["generate"].get("labels") == row["sample_label_path"],
        "data.label_path": config["data"].get("label_path") == row["train_label_paths"],
    }
    if row.get("normalization_mode") == "per-field":
        checks.update(
            {
                "data.normalization.per_field": config["data"].get("normalization") == ["tanh"] * row["num_classes"],
                "data.norm_kwargs.per_field": isinstance(config["data"].get("norm_kwargs"), list)
                and len(config["data"].get("norm_kwargs")) == row["num_classes"],
                "data.transform.per_field": config["data"].get("transform") == [["log"]] * row["num_classes"],
            }
        )
    else:
        checks.update(
            {
                "data.normalization.shared": config["data"].get("normalization") == "tanh",
                "data.norm_kwargs.shared": isinstance(config["data"].get("norm_kwargs"), dict),
                "data.transform.shared": config["data"].get("transform") == ["log"],
            }
        )
    failed.extend(name for name, ok in checks.items() if not ok)
    for field, class_id, arr in zip(row["fields"], range(row["num_classes"]), labels):
        if not np.issubdtype(arr.dtype, np.integer):
            failed.append(f"{field}.labels.integer")
        if arr.shape != (row["n_train_simulations_per_field"],):
            failed.append(f"{field}.labels.shape")
        if len(arr) and not np.all(arr == class_id):
            failed.append(f"{field}.labels.class_id")

    sample_labels = np.load(row["sample_label_path"])
    if not np.issubdtype(sample_labels.dtype, np.integer):
        failed.append("sample.labels.integer")
    if sample_labels.shape != (row["sample_n"],):
        failed.append("sample.labels.shape")
    if len(sample_labels) and (sample_labels.min() < 0 or sample_labels.max() >= row["num_classes"]):
        failed.append("sample.labels.range")
    if failed:
        raise ValueError(f"{config_path} failed checks: {', '.join(failed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="CAMELS IllustrisTNG 3d_grids directory.")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT, help="Root for saved checkpoints.")
    parser.add_argument(
        "--fields",
        default=os.environ.get("NF_CLASS_FIELDS", ",".join(FIELDS)),
        help="Comma-separated field classes.",
    )
    parser.add_argument("--n-train-sims", type=int, default=env_int("NF_CLASS_N_TRAIN_SIMS", N_TRAIN_SIMS))
    parser.add_argument("--sample-n", type=int, default=env_int("NF_CLASS_SAMPLE_N", SAMPLE_N))
    parser.add_argument("--target-updates", type=int, default=env_int("NF_CLASS_TARGET_UPDATES", TARGET_UPDATES))
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=env_int("NF_CLASS_CHECKPOINT_EVERY_UPDATES", CHECKPOINT_EVERY_UPDATES),
    )
    parser.add_argument(
        "--normalization-mode",
        default=os.environ.get("NF_CLASS_NORMALIZATION_MODE", "shared"),
        choices=["shared", "per-field", "per_field"],
        help="Use one shared normalization, or one fitted normalization per field class.",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate existing config/labels without writing.")
    parser.add_argument("--print-runs", action="store_true", help="Print run name and exit.")
    parser.add_argument("--print-table", action="store_true", help="Print one-line run table and exit.")
    parser.add_argument("--check-files", action="store_true", help="Require all grid files to exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    fields = parse_fields(args.fields)
    normalization_mode = normalize_mode(args.normalization_mode)
    name = run_name(fields, args.n_train_sims, normalization_mode, args.target_updates)

    if args.print_runs:
        print(name)
        return

    config_path = project_dir / "local" / SWEEP_NAME / "configs" / f"{name}.yaml"
    manifest_path = project_dir / "local" / SWEEP_NAME / "manifest.json"

    if args.check_only:
        with manifest_path.open() as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = [rows]
        matches = [row for row in rows if row.get("run_name") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one manifest row for {name}, got {len(matches)}.")
        assert_config(config_path, matches[0])
        print(f"Validated {SWEEP_NAME} config.")
        return

    if args.check_files:
        missing = [image_path(data_root, field) for field in fields if not image_path(data_root, field).exists()]
        if missing:
            raise FileNotFoundError("Missing grid files:\n" + "\n".join(str(path) for path in missing))

    label_paths = write_labels(
        project_dir=project_dir,
        fields=fields,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        normalization_mode=normalization_mode,
        target_updates=args.target_updates,
    )
    config = build_config(
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        fields=fields,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        label_paths=label_paths,
        normalization_mode=normalization_mode,
        target_updates=args.target_updates,
        checkpoint_every_updates=args.checkpoint_every_updates,
    )
    row = manifest_row(
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        fields=fields,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        label_paths=label_paths,
        normalization_mode=normalization_mode,
        target_updates=args.target_updates,
        checkpoint_every_updates=args.checkpoint_every_updates,
    )

    if args.print_table:
        cols = [
            "run_name",
            "fields",
            "normalization_mode",
            "dataset_size",
            "steps_per_epoch",
            "epochs",
            "actual_updates",
            "conditioning",
            "num_classes",
        ]
        print("\t".join(cols))
        print("\t".join(str(row[col]) for col in cols))
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote {config_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with ManifestLock(manifest_path.with_suffix(".lock")):
        existing_rows: list[dict[str, Any]] = []
        if manifest_path.exists() and manifest_path.stat().st_size > 0:
            try:
                with manifest_path.open() as f:
                    loaded = json.load(f)
                existing_rows = [loaded] if isinstance(loaded, dict) else list(loaded)
            except json.JSONDecodeError:
                existing_rows = []
        rows = [existing for existing in existing_rows if existing.get("run_name") != row["run_name"]]
        rows.append(row)
        tmp_path = manifest_path.with_suffix(f".{os.getpid()}.tmp")
        with tmp_path.open("w") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        tmp_path.replace(manifest_path)
    print(f"Wrote {manifest_path}")
    print(f"Wrote labels under {project_dir / 'local' / SWEEP_NAME / 'labels'}")


if __name__ == "__main__":
    main()
