#!/usr/bin/env python
"""Prepare ten clean DiT-L16 runs that each target 300k optimizer updates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_nf_generalize_fig2_dit_configs as base


SWEEP_NAME = "nf_generalize_fig2_dit_l16_fresh300k_v2"
MANIFEST_VERSION = 1
TRAINING_SEED = 123
TARGET_TOTAL_UPDATES = 300_000
DEFAULT_SAFETY_CHECKPOINT_UPDATES = 5_000
SAMPLE_LABEL = "dpm50_fresh300k_v2"
DEFAULT_CHECKPOINT_ROOT = Path(
    f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
)


def fresh_run_name(dataset_tag: str) -> str:
    return f"nf_fig2_dit_l16_{dataset_tag}_noaug_fresh300k_v2_seed123"


def selected_base_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    tags = set(args.dataset_tag or [tag for tag, _size in base.RUN_SIZES])
    names = set(args.run_name or [])
    rows = []
    for source in base.iter_runs():
        if source["arch"] != "dit_l16" or source["dataset_tag"] not in tags:
            continue
        row = deepcopy(source)
        row["run_name"] = fresh_run_name(source["dataset_tag"])
        if names and row["run_name"] not in names:
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: int(row["dataset_size"]))


def fresh_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoint_root = Path(args.checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
    safety_updates = int(args.safety_checkpoint_updates)
    if safety_updates <= 0:
        raise ValueError("Safety checkpoint updates must be positive")

    rows: list[dict[str, Any]] = []
    for source in selected_base_rows(args):
        run_name = source["run_name"]
        steps_per_epoch = int(source["optimizer_steps_per_epoch"])
        final_num_epochs = math.ceil(TARGET_TOTAL_UPDATES / steps_per_epoch)
        expected_final_epoch = final_num_epochs - 1
        checkpoint_every_epochs = max(1, round(safety_updates / steps_per_epoch))
        checkpoint_dir = checkpoint_root / f"{run_name}_checkpoints"
        config_rel = f"local/{SWEEP_NAME}/configs/{run_name}.yaml"

        row = deepcopy(source)
        row.update(
            {
                "manifest_version": MANIFEST_VERSION,
                "sweep_name": SWEEP_NAME,
                "arch": "dit_l16",
                "arch_label": "DiT-L16 fresh 300k v2",
                "run_name": run_name,
                "fresh_initialization": True,
                "training_seed": TRAINING_SEED,
                "target_total_updates": TARGET_TOTAL_UPDATES,
                "optimizer_steps_per_epoch": steps_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "final_num_epochs": final_num_epochs,
                "expected_final_epoch": expected_final_epoch,
                "actual_total_updates": final_num_epochs * steps_per_epoch,
                "checkpoint_every_target_updates": safety_updates,
                "checkpoint_every_n_epochs": checkpoint_every_epochs,
                "checkpoint_every_actual_updates": checkpoint_every_epochs
                * steps_per_epoch,
                "checkpoint_dir": str(checkpoint_dir),
                "expected_checkpoint": str(
                    checkpoint_dir / f"checkpoint-epoch-{expected_final_epoch:04d}"
                ),
                "sample_label": SAMPLE_LABEL,
                "sample_path": (
                    f"results/{SWEEP_NAME}/samples/"
                    f"{run_name}_seed{{seed}}_{{sample_label}}.npz"
                ),
                "config": config_rel,
                "note": (
                    "Clean DiT-L16 initialization from seed 123. Train directly "
                    "to 300k requested optimizer updates with resumable recovery "
                    "checkpoints and no dependency on the failed staged sweep."
                ),
            }
        )
        rows.append(row)
    return rows


def build_config(row: dict[str, Any]) -> dict[str, Any]:
    config = base.build_config(
        row["run_name"],
        "dit_l16",
        deepcopy(row["source_counts"]),
        int(row["dataset_size"]),
    )
    config["io"]["output_dir"] = row["checkpoint_dir"]
    config["train"]["num_epochs"] = int(row["final_num_epochs"])
    config["train"]["checkpoint_every_n_epochs"] = int(
        row["checkpoint_every_n_epochs"]
    )
    return config


def assert_config(path: Path, row: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen config: {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    expected = build_config(row)
    for key in (
        "io",
        "model",
        "data",
        "noise_scheduler",
        "optimizer",
        "lr_scheduler",
        "train",
    ):
        if config.get(key) != expected.get(key):
            raise ValueError(f"{path} has a mismatched {key} section")


def manifest_path(project_dir: Path) -> Path:
    return project_dir / "local" / SWEEP_NAME / "manifest.json"


def analysis_manifest_path(project_dir: Path) -> Path:
    return project_dir / "local" / SWEEP_NAME / "analysis_manifest.json"


def load_existing_rows(project_dir: Path) -> list[dict[str, Any]]:
    path = manifest_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen manifest: {path}")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError(f"Frozen manifest must contain ten rows; found {len(rows)}")
    for index, row in enumerate(rows):
        if row.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(f"Manifest version mismatch at row {index}")
        if int(row.get("target_total_updates", -1)) != TARGET_TOTAL_UPDATES:
            raise ValueError(f"Target update mismatch at row {index}")
        if "fresh300k_v2" not in row.get("run_name", ""):
            raise ValueError(f"Run identity mismatch at row {index}")
    return rows


def filter_rows(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.run_name:
        names = set(args.run_name)
        rows = [row for row in rows if row["run_name"] in names]
    if args.dataset_tag:
        tags = set(args.dataset_tag)
        rows = [row for row in rows if row["dataset_tag"] in tags]
    return sorted(rows, key=lambda row: int(row["dataset_size"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--dataset-tag", action="append")
    parser.add_argument("--run-name", action="append")
    parser.add_argument(
        "--safety-checkpoint-updates",
        type=int,
        default=DEFAULT_SAFETY_CHECKPOINT_UPDATES,
    )
    parser.add_argument("--use-existing-manifest", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-runs", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    parser.add_argument(
        "--print-field",
        choices=(
            "config",
            "checkpoint_dir",
            "expected_checkpoint",
            "sample_label",
            "sample_path",
            "target_total_updates",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    all_rows = (
        load_existing_rows(project_dir)
        if args.use_existing_manifest
        else fresh_rows(args)
    )
    rows = filter_rows(all_rows, args)

    if args.print_runs:
        for row in rows:
            print(row["run_name"])
        return
    if args.print_field:
        if len(rows) != 1:
            raise SystemExit(
                f"--print-field requires exactly one selected row; found {len(rows)}"
            )
        print(rows[0][args.print_field])
        return
    if args.print_table:
        columns = (
            "run_name",
            "dataset_size",
            "target_total_updates",
            "actual_total_updates",
            "expected_final_epoch",
            "checkpoint_every_actual_updates",
        )
        print("\t".join(columns))
        for row in rows:
            print("\t".join(str(row[column]) for column in columns))
        return
    if args.check_only:
        for row in rows:
            assert_config(project_dir / row["config"], row)
        print(f"Validated {len(rows)} frozen {SWEEP_NAME} configs.")
        return

    for row in rows:
        path = project_dir / row["config"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            yaml.safe_dump(build_config(row), handle, sort_keys=False)
        print(f"Wrote {path}")

    manifest = manifest_path(project_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(all_rows, indent=2) + "\n")
    analysis_manifest_path(project_dir).write_text(
        json.dumps(all_rows, indent=2) + "\n"
    )
    print(f"Wrote {manifest}")
    print(f"Wrote {analysis_manifest_path(project_dir)}")
    print("Fresh sweep: 10 independent DiT-L16 runs, final target 300000 updates.")


if __name__ == "__main__":
    main()
