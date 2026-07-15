#!/usr/bin/env python
"""Prepare resumable 25k-update continuation stages for small-data DiT-L16 runs.

The generated manifest freezes exact base, previous-stage, and target
checkpoints. Original training directories are read-only inputs; continuation
stages write to isolated directories so overshot historical checkpoints cannot
change resume selection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_nf_generalize_fig2_dit_configs as base


CONTINUE_SWEEP_NAME = "nf_generalize_fig2_dit_l16_continue"
MANIFEST_VERSION = 2
DEFAULT_DATASET_TAGS = ("d2p06", "d2p07", "d2p08", "d2p09", "d2p10")
DEFAULT_STAGE_UPDATES = 25_000
DEFAULT_STAGES = 4
DEFAULT_SAFETY_CHECKPOINT_UPDATES = 5_000
DEFAULT_CONTINUATION_CHECKPOINT_ROOT = Path(
    "/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_generalize_fig2_dit_l16_continue"
)
CHECKPOINT_RE = re.compile(r"checkpoint-epoch-(\d+)$")


def checkpoint_epoch(path: Path) -> int | None:
    match = CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else None


def latest_checkpoint_epoch(checkpoint_dir: Path) -> int | None:
    if not checkpoint_dir.exists():
        return None
    epochs = [
        epoch
        for epoch in (checkpoint_epoch(path) for path in checkpoint_dir.glob("checkpoint-epoch-*"))
        if epoch is not None
    ]
    return max(epochs) if epochs else None


def _original_checkpoint_dir(row: dict[str, Any], checkpoint_root: Path | None) -> Path:
    if checkpoint_root is None:
        return Path(row["checkpoint_dir"])
    return checkpoint_root / f'{row["run_name"]}_checkpoints'


def _continuation_checkpoint_dir(
    row: dict[str, Any], continuation_checkpoint_root: Path | None
) -> Path:
    root = continuation_checkpoint_root or DEFAULT_CONTINUATION_CHECKPOINT_ROOT
    return root / f'{row["run_name"]}_checkpoints'


def _selected_base_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    tags = set(args.dataset_tag or DEFAULT_DATASET_TAGS)
    names = set(args.run_name or [])
    rows = [
        row
        for row in base.iter_runs()
        if row["arch"] == "dit_l16"
        and row["dataset_tag"] in tags
        and (not names or row["run_name"] in names)
    ]
    return sorted(rows, key=lambda row: int(row["dataset_size"]))


def continue_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return frozen stage arithmetic for the selected DiT-L16 runs."""
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root is not None else None
    continuation_root_arg = getattr(args, "continuation_checkpoint_root", None)
    continuation_checkpoint_root = (
        Path(continuation_root_arg) if continuation_root_arg is not None else None
    )
    stage_updates = int(args.stage_updates)
    stages = int(args.stages)
    safety_updates = int(args.safety_checkpoint_updates)
    rows: list[dict[str, Any]] = []

    for base_row in _selected_base_rows(args):
        original_checkpoint_dir = _original_checkpoint_dir(base_row, checkpoint_root)
        checkpoint_dir = _continuation_checkpoint_dir(
            base_row, continuation_checkpoint_root
        )
        base_checkpoint_epoch = int(base_row["epochs"]) - 1
        base_checkpoint = (
            original_checkpoint_dir / f"checkpoint-epoch-{base_checkpoint_epoch:04d}"
        )
        if not base_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Missing exact 200k checkpoint for {base_row['run_name']}: {base_checkpoint}"
            )

        steps_per_epoch = int(base_row["optimizer_steps_per_epoch"])
        safety_epochs = max(1, round(safety_updates / steps_per_epoch))

        for stage in range(1, stages + 1):
            previous_cumulative_updates = (stage - 1) * stage_updates
            cumulative_target_updates = stage * stage_updates
            previous_cumulative_epochs = math.ceil(
                previous_cumulative_updates / steps_per_epoch
            )
            cumulative_epochs = max(1, math.ceil(cumulative_target_updates / steps_per_epoch))
            previous_epoch = base_checkpoint_epoch + previous_cumulative_epochs
            final_epoch = base_checkpoint_epoch + cumulative_epochs
            final_num_epochs = final_epoch + 1
            stage_additional_epochs = final_epoch - previous_epoch
            stage_target_total_updates = int(base_row["target_updates"]) + cumulative_target_updates
            sample_label = f"dpm50_cont_{stage_target_total_updates // 1000}k"
            config_rel = (
                f"local/{CONTINUE_SWEEP_NAME}/configs/stage_{stage}/{base_row['run_name']}.yaml"
            )

            row = deepcopy(base_row)
            row.update(
                {
                    "manifest_version": MANIFEST_VERSION,
                    "continue_sweep_name": CONTINUE_SWEEP_NAME,
                    "continue_stage": stage,
                    "original_checkpoint_dir": str(original_checkpoint_dir),
                    "base_checkpoint_epoch": base_checkpoint_epoch,
                    "base_checkpoint": str(base_checkpoint),
                    "previous_expected_epoch": previous_epoch,
                    "previous_expected_checkpoint": str(
                        checkpoint_dir / f"checkpoint-epoch-{previous_epoch:04d}"
                    ),
                    "stage_target_updates": stage_updates,
                    "stage_additional_epochs": stage_additional_epochs,
                    "stage_actual_updates": stage_additional_epochs * steps_per_epoch,
                    "cumulative_target_updates": cumulative_target_updates,
                    "cumulative_actual_updates": cumulative_epochs * steps_per_epoch,
                    "target_total_updates": stage_target_total_updates,
                    "final_num_epochs": final_num_epochs,
                    "expected_final_epoch": final_epoch,
                    "checkpoint_every_target_updates": safety_updates,
                    "checkpoint_every_n_epochs": safety_epochs,
                    "checkpoint_every_actual_updates": safety_epochs * steps_per_epoch,
                    "checkpoint_dir": str(checkpoint_dir),
                    "expected_checkpoint": str(checkpoint_dir / f"checkpoint-epoch-{final_epoch:04d}"),
                    "sample_label": sample_label,
                    "sample_path": (
                        f"results/{base.SWEEP_NAME}/samples/"
                        f"{base_row['run_name']}_seed{{seed}}_{sample_label}.npz"
                    ),
                    "config": config_rel,
                    "note": (
                        "DiT-L16 controlled continuation: fixed 25k-update stage with "
                        "approximately 5k-update recovery checkpoints."
                    ),
                }
            )
            rows.append(row)
    return rows


def build_continue_config(row: dict[str, Any]) -> dict[str, Any]:
    config = base.build_config(
        row["run_name"],
        "dit_l16",
        deepcopy(row["source_counts"]),
        int(row["dataset_size"]),
    )
    config["io"]["output_dir"] = row["checkpoint_dir"]
    config["train"]["num_epochs"] = int(row["final_num_epochs"])
    config["train"]["checkpoint_every_n_epochs"] = int(row["checkpoint_every_n_epochs"])
    return config


def assert_config(path: Path, row: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)

    expected = base.build_config(
        row["run_name"],
        "dit_l16",
        deepcopy(row["source_counts"]),
        int(row["dataset_size"]),
    )
    checks = {
        "io.output_dir": config["io"].get("output_dir") == row["checkpoint_dir"],
        "model.class": config["model"].get("class") == "DiTTransformer2DModel",
        "model.kwargs": config["model"].get("kwargs") == expected["model"]["kwargs"],
        "data.constant_label": config["data"].get("constant_label") == 0,
        "data.no_augmentation": "augmentations" not in config["data"],
        "noise_scheduler": config["noise_scheduler"] == expected["noise_scheduler"],
        "optimizer": config["optimizer"] == expected["optimizer"],
        "lr_scheduler": config["lr_scheduler"] == expected["lr_scheduler"],
        "train.num_epochs": config["train"].get("num_epochs") == row["final_num_epochs"],
        "train.batch_size": config["train"].get("batch_size") == row["batch_size"],
        "train.gradient_accumulation_steps": (
            config["train"].get("gradient_accumulation_steps")
            == row["gradient_accumulation_steps"]
        ),
        "train.checkpoint_every_n_epochs": (
            config["train"].get("checkpoint_every_n_epochs")
            == row["checkpoint_every_n_epochs"]
        ),
        "train.ema_sigma_rels": config["train"].get("ema_sigma_rels") == row["ema_sigma_rels"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed continuation checks: {', '.join(failed)}")


def _manifest_path(project_dir: Path) -> Path:
    return project_dir / "local" / CONTINUE_SWEEP_NAME / "manifest.json"


def _load_existing_rows(project_dir: Path) -> list[dict[str, Any]]:
    path = _manifest_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen continuation manifest: {path}. Run this script once without --use-existing-manifest."
        )
    rows = json.loads(path.read_text())
    required = {
        "manifest_version",
        "base_checkpoint",
        "previous_expected_checkpoint",
        "expected_checkpoint",
    }
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Frozen continuation manifest version is invalid or empty: {path}")
    for index, row in enumerate(rows):
        if row.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(
                f"Frozen continuation manifest version mismatch at row {index}: "
                f"expected {MANIFEST_VERSION}, got {row.get('manifest_version')!r}. "
                "Regenerate it without --use-existing-manifest."
            )
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"Frozen continuation manifest version {MANIFEST_VERSION} row {index} "
                f"is missing: {', '.join(missing)}"
            )
    return rows


def seed_continuation_directories(rows: list[dict[str, Any]]) -> None:
    """Seed each isolated continuation directory with its exact 200k checkpoint."""
    seen: set[tuple[Path, Path]] = set()
    for row in rows:
        source = Path(row["base_checkpoint"])
        checkpoint_dir = Path(row["checkpoint_dir"])
        key = (checkpoint_dir, source)
        if key in seen:
            continue
        seen.add(key)

        if not source.is_dir():
            raise FileNotFoundError(f"Missing exact base checkpoint: {source}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        seed = checkpoint_dir / source.name
        if seed.exists() or seed.is_symlink():
            if seed.is_symlink() and seed.resolve() == source.resolve():
                continue
            raise ValueError(
                f"Continuation directory contains a conflicting checkpoint at {seed}"
            )
        seed.symlink_to(source.resolve(), target_is_directory=True)


def _filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.stage is not None:
        rows = [row for row in rows if int(row["continue_stage"]) == int(args.stage)]
    if args.run_name:
        names = set(args.run_name)
        rows = [row for row in rows if row["run_name"] in names]
    if args.dataset_tag:
        tags = set(args.dataset_tag)
        rows = [row for row in rows if row["dataset_tag"] in tags]
    return sorted(rows, key=lambda row: (int(row["continue_stage"]), int(row["dataset_size"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--continuation-checkpoint-root", type=Path)
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
    parser.add_argument("--seed-checkpoints", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-runs", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    parser.add_argument(
        "--print-field",
        choices=(
            "config",
            "checkpoint_dir",
            "base_checkpoint",
            "previous_expected_checkpoint",
            "expected_checkpoint",
            "sample_label",
            "sample_path",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    if args.use_existing_manifest:
        rows = _filter_rows(_load_existing_rows(project_dir), args)
    else:
        rows = _filter_rows(continue_rows(args), args)

    if args.print_runs:
        for row in rows:
            print(row["run_name"])
        return

    if args.seed_checkpoints:
        seed_continuation_directories(rows)
        print(f"Seeded {len({row['run_name'] for row in rows})} isolated continuation directories.")
        return

    if args.print_field:
        if len(rows) != 1:
            raise SystemExit(
                f"--print-field requires exactly one row after filtering; selected {len(rows)}"
            )
        print(rows[0][args.print_field])
        return

    if args.print_table:
        columns = (
            "manifest_version",
            "continue_stage",
            "run_name",
            "dataset_size",
            "steps_per_epoch",
            "base_checkpoint_epoch",
            "previous_expected_epoch",
            "stage_additional_epochs",
            "final_num_epochs",
            "expected_final_epoch",
            "cumulative_target_updates",
            "checkpoint_every_actual_updates",
            "sample_label",
        )
        print("\t".join(columns))
        for row in rows:
            print("\t".join(str(row[column]) for column in columns))
        return

    if args.check_only:
        for row in rows:
            assert_config(project_dir / row["config"], row)
        print(f"Validated {len(rows)} frozen {CONTINUE_SWEEP_NAME} configs.")
        return

    config_root = project_dir / "local" / CONTINUE_SWEEP_NAME / "configs"
    for row in rows:
        path = project_dir / row["config"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            yaml.safe_dump(build_continue_config(row), handle, sort_keys=False)
        print(f"Wrote {path}")

    manifest_path = _manifest_path(project_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {manifest_path}")

    # The analyzers must see the same five runs for every checkpoint label.
    # Keeping this separate from the 20-row stage manifest prevents duplicate
    # run rows and ensures PCA is fitted on the same reference selection.
    analysis_manifest_path = manifest_path.parent / "analysis_manifest.json"
    analysis_rows = _selected_base_rows(args)
    analysis_manifest_path.write_text(json.dumps(analysis_rows, indent=2) + "\n")
    print(f"Wrote {analysis_manifest_path}")
    print(f"Recovery checkpoints are approximately every {args.safety_checkpoint_updates:,} updates.")


if __name__ == "__main__":
    main()
