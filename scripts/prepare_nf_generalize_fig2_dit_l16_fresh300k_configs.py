#!/usr/bin/env python
"""Prepare a fresh ten-size DiT-L16 sweep through 300k optimizer updates."""

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


SWEEP_NAME = "nf_generalize_fig2_dit_l16_fresh300k"
MANIFEST_VERSION = 1
TRAINING_SEED = 123
DEFAULT_STAGE_UPDATES = 25_000
DEFAULT_STAGES = 12
DEFAULT_SAFETY_CHECKPOINT_UPDATES = 5_000
SCIENTIFIC_UPDATES = frozenset({200_000, 225_000, 250_000, 275_000, 300_000})
DEFAULT_CHECKPOINT_ROOT = Path(
    f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
)


def fresh_run_name(dataset_tag: str) -> str:
    return f"nf_fig2_dit_l16_{dataset_tag}_noaug_fresh300k_seed123"


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
    stage_updates = int(args.stage_updates)
    stages = int(args.stages)
    safety_updates = int(args.safety_checkpoint_updates)
    if stage_updates <= 0 or stages <= 0 or safety_updates <= 0:
        raise ValueError("stage, stage count, and safety checkpoint updates must be positive")
    if stage_updates * stages != 300_000:
        raise ValueError(
            f"Fresh DiT-L16 sweep must end at 300000 updates; got {stage_updates * stages}"
        )

    rows: list[dict[str, Any]] = []
    for source in selected_base_rows(args):
        run_name = source["run_name"]
        checkpoint_dir = checkpoint_root / f"{run_name}_checkpoints"
        steps_per_epoch = int(source["optimizer_steps_per_epoch"])
        checkpoint_every_epochs = max(1, round(safety_updates / steps_per_epoch))

        for stage in range(1, stages + 1):
            previous_target_updates = (stage - 1) * stage_updates
            target_total_updates = stage * stage_updates
            previous_epochs = math.ceil(previous_target_updates / steps_per_epoch)
            final_num_epochs = math.ceil(target_total_updates / steps_per_epoch)
            previous_epoch = previous_epochs - 1
            expected_final_epoch = final_num_epochs - 1
            scientific = target_total_updates in SCIENTIFIC_UPDATES
            sample_label = (
                f"dpm50_fresh_{target_total_updates // 1000}k" if scientific else None
            )
            config_rel = (
                f"local/{SWEEP_NAME}/configs/stage_{stage:02d}/{run_name}.yaml"
            )
            expected_checkpoint = (
                checkpoint_dir / f"checkpoint-epoch-{expected_final_epoch:04d}"
            )
            previous_checkpoint = (
                checkpoint_dir / f"checkpoint-epoch-{previous_epoch:04d}"
                if previous_epoch >= 0
                else None
            )
            row = deepcopy(source)
            row.update(
                {
                    "manifest_version": MANIFEST_VERSION,
                    "sweep_name": SWEEP_NAME,
                    "stage": stage,
                    "arch": "dit_l16",
                    "arch_label": "DiT-L16 fresh 300k",
                    "run_name": run_name,
                    "fresh_initialization": True,
                    "training_seed": TRAINING_SEED,
                    "stage_target_updates": stage_updates,
                    "previous_target_updates": previous_target_updates,
                    "target_total_updates": target_total_updates,
                    "optimizer_steps_per_epoch": steps_per_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "previous_expected_epoch": previous_epoch,
                    "expected_final_epoch": expected_final_epoch,
                    "final_num_epochs": final_num_epochs,
                    "stage_additional_epochs": final_num_epochs - previous_epochs,
                    "actual_total_updates": final_num_epochs * steps_per_epoch,
                    "checkpoint_every_target_updates": safety_updates,
                    "checkpoint_every_n_epochs": checkpoint_every_epochs,
                    "checkpoint_every_actual_updates": (
                        checkpoint_every_epochs * steps_per_epoch
                    ),
                    "checkpoint_dir": str(checkpoint_dir),
                    "previous_expected_checkpoint": (
                        str(previous_checkpoint) if previous_checkpoint else None
                    ),
                    "expected_checkpoint": str(expected_checkpoint),
                    "scientific_checkpoint": scientific,
                    "sample_label": sample_label,
                    "sample_path": (
                        f"results/{SWEEP_NAME}/samples/"
                        f"{run_name}_seed{{seed}}_{{sample_label}}.npz"
                    ),
                    "config": config_rel,
                    "note": (
                        "Fresh DiT-L16 run from seed 123; all ten dataset sizes "
                        "train through 300k requested optimizer updates."
                    ),
                }
            )
            rows.append(row)
    return rows


def build_stage_config(row: dict[str, Any]) -> dict[str, Any]:
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


def assert_stage_config(path: Path, row: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen stage config: {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    expected = build_stage_config(row)
    checks = {
        "io.output_dir": config["io"].get("output_dir") == row["checkpoint_dir"],
        "model": config.get("model") == expected["model"],
        "data": config.get("data") == expected["data"],
        "noise_scheduler": config.get("noise_scheduler") == expected["noise_scheduler"],
        "optimizer": config.get("optimizer") == expected["optimizer"],
        "lr_scheduler": config.get("lr_scheduler") == expected["lr_scheduler"],
        "train": config.get("train") == expected["train"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed fresh sweep checks: {', '.join(failed)}")


def manifest_path(project_dir: Path) -> Path:
    return project_dir / "local" / SWEEP_NAME / "manifest.json"


def analysis_manifest_path(project_dir: Path) -> Path:
    return project_dir / "local" / SWEEP_NAME / "analysis_manifest.json"


def load_existing_rows(project_dir: Path) -> list[dict[str, Any]]:
    path = manifest_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen fresh-sweep manifest: {path}")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(
            f"Frozen fresh-sweep manifest must be a list; found {type(rows).__name__}"
        )
    for index, row in enumerate(rows):
        if row.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(
                f"Frozen manifest version mismatch at row {index}: "
                f"expected {MANIFEST_VERSION}, got {row.get('manifest_version')!r}"
            )
    if len(rows) != 120:
        raise ValueError(
            f"Frozen fresh-sweep manifest must contain 120 rows; found {len(rows)}"
        )
    required = {
        "manifest_version",
        "stage",
        "training_seed",
        "fresh_initialization",
        "previous_expected_checkpoint",
        "expected_checkpoint",
        "target_total_updates",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Frozen manifest row {index} is missing {sorted(missing)}")
    return rows


def filter_rows(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.stage is not None:
        rows = [row for row in rows if int(row["stage"]) == int(args.stage)]
    if args.run_name:
        names = set(args.run_name)
        rows = [row for row in rows if row["run_name"] in names]
    if args.dataset_tag:
        tags = set(args.dataset_tag)
        rows = [row for row in rows if row["dataset_tag"] in tags]
    return sorted(rows, key=lambda row: (int(row["stage"]), int(row["dataset_size"])))


def analysis_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["run_name"] in by_run:
            continue
        item = deepcopy(row)
        item["sample_path"] = (
            f"results/{SWEEP_NAME}/samples/"
            f"{row['run_name']}_seed{{seed}}_{{sample_label}}.npz"
        )
        by_run[row["run_name"]] = item
    return sorted(by_run.values(), key=lambda row: int(row["dataset_size"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--dataset-tag", action="append")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--stage", type=int, choices=range(1, DEFAULT_STAGES + 1))
    parser.add_argument("--stage-updates", type=int, default=DEFAULT_STAGE_UPDATES)
    parser.add_argument("--stages", type=int, default=DEFAULT_STAGES)
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
            "previous_expected_checkpoint",
            "expected_checkpoint",
            "scientific_checkpoint",
            "sample_label",
            "sample_path",
            "target_total_updates",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    if args.use_existing_manifest:
        all_rows = load_existing_rows(project_dir)
    else:
        all_rows = fresh_rows(args)
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
        value = rows[0][args.print_field]
        print("" if value is None else value)
        return
    if args.print_table:
        columns = (
            "stage",
            "run_name",
            "dataset_size",
            "target_total_updates",
            "actual_total_updates",
            "expected_final_epoch",
            "scientific_checkpoint",
            "sample_label",
        )
        print("\t".join(columns))
        for row in rows:
            print("\t".join(str(row[column]) for column in columns))
        return
    if args.check_only:
        for row in rows:
            assert_stage_config(project_dir / row["config"], row)
        print(f"Validated {len(rows)} frozen {SWEEP_NAME} stage configs.")
        return

    for row in rows:
        path = project_dir / row["config"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            yaml.safe_dump(build_stage_config(row), handle, sort_keys=False)
        print(f"Wrote {path}")

    path = manifest_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_rows, indent=2) + "\n")
    analysis_manifest_path(project_dir).write_text(
        json.dumps(analysis_rows(all_rows), indent=2) + "\n"
    )
    print(f"Wrote {path}")
    print(f"Wrote {analysis_manifest_path(project_dir)}")
    print(
        "Fresh sweep: 10 DiT-L16 runs, seed 123, 12 stages, final target 300000 updates."
    )


if __name__ == "__main__":
    main()
