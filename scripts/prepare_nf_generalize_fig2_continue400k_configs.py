#!/usr/bin/env python
"""Prepare continuation configs for selected Fig. 2 CAMELS runs.

These configs intentionally reuse the original nf_generalize_fig2 checkpoint
directories, but raise train.num_epochs so cosmodiff_train resumes from the
latest checkpoint instead of starting new runs.

Default continuation set:
- u64 d2p15, to check whether the largest u64 run improves with more updates
- u128 d2p06..d2p13, to extend completed u128 runs

The still-running u128 d2p14/d2p15 jobs are excluded by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

import prepare_nf_generalize_fig2_configs as base


CONTINUE_SWEEP_NAME = "nf_generalize_fig2_continue400k"
TOTAL_TARGET_UPDATES = 400_000
CHECKPOINT_EVERY_UPDATES = 20_000

DEFAULT_SELECTION = {
    ("u64", "d2p15"),
    ("u128", "d2p06"),
    ("u128", "d2p07"),
    ("u128", "d2p08"),
    ("u128", "d2p09"),
    ("u128", "d2p10"),
    ("u128", "d2p11"),
    ("u128", "d2p12"),
    ("u128", "d2p13"),
}

PENDING_U128_SELECTION = {
    ("u128", "d2p14"),
    ("u128", "d2p15"),
}


def epochs_for_total_updates(dataset_size: int, batch_size: int, total_updates: int) -> int:
    steps_per_epoch = base.steps_per_epoch(dataset_size, batch_size)
    return max(1, math.ceil(int(total_updates) / steps_per_epoch))


def checkpoint_epochs_for(dataset_size: int, batch_size: int) -> int:
    steps_per_epoch = base.steps_per_epoch(dataset_size, batch_size)
    return max(1, round(CHECKPOINT_EVERY_UPDATES / steps_per_epoch))


def selected_pairs(args: argparse.Namespace) -> set[tuple[str, str]]:
    pairs = set(DEFAULT_SELECTION)
    if args.include_u128_pending:
        pairs |= PENDING_U128_SELECTION
    if args.run_name or args.arch or args.dataset_tag:
        pairs = {
            (row["arch"], row["dataset_tag"])
            for row in base.iter_runs()
            if (not args.run_name or row["run_name"] in args.run_name)
            and (not args.arch or row["arch"] in args.arch)
            and (not args.dataset_tag or row["dataset_tag"] in args.dataset_tag)
        }
    return pairs


def continue_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs = selected_pairs(args)
    rows: list[dict[str, Any]] = []
    for row in base.iter_runs():
        pair = (row["arch"], row["dataset_tag"])
        if pair not in pairs:
            continue
        batch_size = int(row["batch_size"])
        dataset_size = int(row["dataset_size"])
        steps_per_epoch = base.steps_per_epoch(dataset_size, batch_size)
        continue_epochs = epochs_for_total_updates(dataset_size, batch_size, args.total_updates)
        continue_actual_updates = continue_epochs * steps_per_epoch
        updated = dict(row)
        updated.update(
            {
                "continue_sweep_name": CONTINUE_SWEEP_NAME,
                "base_target_updates": int(row["target_updates"]),
                "continue_target_updates": int(args.total_updates),
                "continue_actual_updates": int(continue_actual_updates),
                "additional_nominal_updates": int(continue_actual_updates - int(row["actual_updates"])),
                "epochs": int(continue_epochs),
                "checkpoint_every_updates": CHECKPOINT_EVERY_UPDATES,
                "checkpoint_every_n_epochs": int(checkpoint_epochs_for(dataset_size, batch_size)),
                "config": f"local/{CONTINUE_SWEEP_NAME}/configs/{row['run_name']}.yaml",
                "checkpoint_dir": row["checkpoint_dir"],
                "note": (
                    "Continuation config: reuse the original checkpoint directory and "
                    f"raise total target updates to about {args.total_updates}."
                ),
            }
        )
        rows.append(updated)
    return rows


def build_continue_config(row: dict[str, Any]) -> dict[str, Any]:
    config = base.build_config(
        row["run_name"],
        row["arch"],
        int(row["dataset_size"]),
        row["source_counts"],
    )
    config["train"]["num_epochs"] = int(row["epochs"])
    config["train"]["checkpoint_every_n_epochs"] = int(row["checkpoint_every_n_epochs"])
    return config


def assert_config(path: Path, row: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)
    checks = {
        "io.output_dir": config["io"]["output_dir"] == row["checkpoint_dir"],
        "train.num_epochs": config["train"].get("num_epochs") == row["epochs"],
        "train.checkpoint_every_n_epochs": (
            config["train"].get("checkpoint_every_n_epochs") == row["checkpoint_every_n_epochs"]
        ),
        "data.no_augmentation": "augmentations" not in config,
        "data.target_size": sum(config["data"]["n_samples"]) * base.SLICES_PER_VOLUME == row["dataset_size"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--check-only", action="store_true", help="Validate existing configs without writing.")
    parser.add_argument("--print-runs", action="store_true", help="Print selected run names and exit.")
    parser.add_argument("--print-table", action="store_true", help="Print selected continuation rows and exit.")
    parser.add_argument("--run-name", action="append", help="Optional run name. Repeatable.")
    parser.add_argument("--arch", choices=sorted(base.ARCHES), action="append", help="Restrict architecture.")
    parser.add_argument("--dataset-tag", action="append", help="Restrict dataset tag such as d2p15.")
    parser.add_argument(
        "--total-updates",
        type=int,
        default=TOTAL_TARGET_UPDATES,
        help="Total target optimizer updates after continuation.",
    )
    parser.add_argument(
        "--include-u128-pending",
        action="store_true",
        help="Also include u128 d2p14/d2p15. Default excludes them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_dir = project_dir / "local" / CONTINUE_SWEEP_NAME / "configs"
    manifest_path = project_dir / "local" / CONTINUE_SWEEP_NAME / "manifest.json"
    rows = continue_rows(args)

    if args.print_runs:
        for row in rows:
            print(row["run_name"])
        return

    if args.print_table:
        columns = [
            "run_name",
            "arch",
            "dataset_tag",
            "dataset_size",
            "steps_per_epoch",
            "actual_updates",
            "continue_actual_updates",
            "additional_nominal_updates",
            "epochs",
        ]
        print("\t".join(columns))
        for row in rows:
            print("\t".join(str(row[col]) for col in columns))
        return

    if args.check_only:
        for row in rows:
            assert_config(config_dir / f"{row['run_name']}.yaml", row)
        print(f"Validated {len(rows)} {CONTINUE_SWEEP_NAME} configs.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = config_dir / f"{row['run_name']}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(build_continue_config(row), f, sort_keys=False)
        print(f"Wrote {path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
