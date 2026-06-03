#!/usr/bin/env python
"""Prepare continuation configs for the continuous HI bias-probe runs.

The continuation configs intentionally reuse the original checkpoint
directories from ``nf_conditional_bias_probe``.  ``cosmodiff_train.py`` resumes
from the latest checkpoint in ``io.output_dir``; here ``train.num_epochs`` is
set to the number of additional epochs corresponding to the requested
additional optimizer updates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


BASE_SWEEP_NAME = "nf_conditional_bias_probe"
CONTINUE_SWEEP_NAME = "nf_conditional_bias_probe_continue"
DATASET_SIZES = (128, 16_384)
ADDITIONAL_UPDATES = 300_000
CHECKPOINT_EVERY_UPDATES = 20_000


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def ceil_updates_to_epochs(steps_per_epoch: int, updates: int) -> int:
    return max(1, math.ceil(int(updates) / max(1, int(steps_per_epoch))))


def checkpoint_epochs_for(steps_per_epoch: int, checkpoint_every_updates: int) -> int:
    return max(1, round(int(checkpoint_every_updates) / max(1, int(steps_per_epoch))))


def load_base_manifest(project_dir: Path, base_sweep_name: str) -> list[dict[str, Any]]:
    path = project_dir / "local" / base_sweep_name / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing base manifest: {path}")
    with path.open() as f:
        rows = json.load(f)
    if not rows:
        raise RuntimeError(f"Empty base manifest: {path}")
    return rows


def base_config_path(project_dir: Path, base_sweep_name: str, row: dict[str, Any]) -> Path:
    explicit = row.get("config")
    if explicit:
        path = Path(str(explicit))
        if not path.is_absolute():
            path = project_dir / path
        if path.exists():
            return path
    return project_dir / "local" / base_sweep_name / "configs" / f"{row['run_name']}.yaml"


def selected_rows(rows: list[dict[str, Any]], dataset_sizes: list[int]) -> list[dict[str, Any]]:
    want = {int(x) for x in dataset_sizes}
    out = [row for row in rows if int(row.get("dataset_size", -1)) in want]
    found = {int(row["dataset_size"]) for row in out}
    missing = sorted(want - found)
    if missing:
        raise RuntimeError(f"Missing dataset sizes in base manifest: {missing}")
    return sorted(out, key=lambda row: int(row["dataset_size"]))


def build_continue_config(
    project_dir: Path,
    base_sweep_name: str,
    continue_sweep_name: str,
    row: dict[str, Any],
    additional_updates: int,
    checkpoint_every_updates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = base_config_path(project_dir, base_sweep_name, row)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing base config for {row['run_name']}: {config_path}")
    with config_path.open() as f:
        config = yaml.safe_load(f)

    steps_per_epoch = int(row["steps_per_epoch"])
    continue_epochs = ceil_updates_to_epochs(steps_per_epoch, additional_updates)
    continue_actual_updates = continue_epochs * steps_per_epoch
    checkpoint_every_n_epochs = checkpoint_epochs_for(steps_per_epoch, checkpoint_every_updates)

    config["train"]["num_epochs"] = int(continue_epochs)
    config["train"]["checkpoint_every_n_epochs"] = int(checkpoint_every_n_epochs)

    updated = dict(row)
    updated.update(
        {
            "base_sweep_name": base_sweep_name,
            "continue_sweep_name": continue_sweep_name,
            "base_config": str(config_path),
            "base_epochs": int(row.get("epochs", 0)),
            "base_target_updates": int(row.get("target_updates", 0)),
            "additional_target_updates": int(additional_updates),
            "continue_epochs": int(continue_epochs),
            "continue_actual_updates": int(continue_actual_updates),
            "epochs": int(continue_epochs),
            "checkpoint_every_updates": int(checkpoint_every_updates),
            "checkpoint_every_n_epochs": int(checkpoint_every_n_epochs),
            "config": f"local/{continue_sweep_name}/configs/{row['run_name']}.yaml",
            "note": (
                "Continuation config: reuse the original bias-probe checkpoint "
                f"directory and run about {additional_updates} additional optimizer updates."
            ),
        }
    )
    return config, updated


def assert_config(project_dir: Path, continue_sweep_name: str, row: dict[str, Any]) -> None:
    path = project_dir / "local" / continue_sweep_name / "configs" / f"{row['run_name']}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing continuation config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)
    checks = {
        "io.output_dir": config["io"]["output_dir"] == row["checkpoint_dir"],
        "train.num_epochs": config["train"].get("num_epochs") == row["continue_epochs"],
        "train.checkpoint_every_n_epochs": (
            config["train"].get("checkpoint_every_n_epochs") == row["checkpoint_every_n_epochs"]
        ),
        "train.conditioning": config["train"].get("conditioning") == "continuous",
        "generate.conditioning": config["generate"].get("conditioning") == "continuous",
        "model.class": config["model"]["class"] == "UNet2DConditionModel",
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--base-sweep-name", default=BASE_SWEEP_NAME)
    parser.add_argument("--continue-sweep-name", default=CONTINUE_SWEEP_NAME)
    parser.add_argument("--dataset-sizes", default=",".join(str(x) for x in DATASET_SIZES))
    parser.add_argument("--additional-updates", type=int, default=ADDITIONAL_UPDATES)
    parser.add_argument("--checkpoint-every-updates", type=int, default=CHECKPOINT_EVERY_UPDATES)
    parser.add_argument("--print-runs", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    dataset_sizes = parse_int_list(args.dataset_sizes)
    base_rows = selected_rows(load_base_manifest(project_dir, args.base_sweep_name), dataset_sizes)
    items = [
        build_continue_config(
            project_dir,
            args.base_sweep_name,
            args.continue_sweep_name,
            row,
            args.additional_updates,
            args.checkpoint_every_updates,
        )
        for row in base_rows
    ]

    if args.print_runs:
        for _config, row in items:
            print(row["run_name"])
        return

    if args.print_table:
        columns = [
            "run_name",
            "regime",
            "dataset_size",
            "steps_per_epoch",
            "base_epochs",
            "continue_epochs",
            "additional_target_updates",
            "continue_actual_updates",
            "checkpoint_every_n_epochs",
            "checkpoint_dir",
        ]
        print("\t".join(columns))
        for _config, row in items:
            print("\t".join(str(row[col]) for col in columns))
        return

    if args.check_only:
        for _config, row in items:
            assert_config(project_dir, args.continue_sweep_name, row)
        print(f"Validated {len(items)} {args.continue_sweep_name} configs.")
        return

    config_dir = project_dir / "local" / args.continue_sweep_name / "configs"
    manifest_path = project_dir / "local" / args.continue_sweep_name / "manifest.json"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config, row in items:
        path = config_dir / f"{row['run_name']}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"Wrote {path}")
        rows.append(row)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
