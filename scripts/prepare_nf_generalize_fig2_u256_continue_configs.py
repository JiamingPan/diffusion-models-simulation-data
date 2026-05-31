#!/usr/bin/env python
"""Prepare continuation configs for the Fig. 2 UNet-256 high-N runs.

The base ``nf_generalize_fig2`` sweep trained every architecture for about
200k optimizer updates. The u256 high-N models can look undertrained in the
reproducibility/generalizability curves, so this helper writes continuation
configs that resume from the latest checkpoint in the original checkpoint
directory.

Important: nkern/cosmo_diffusion interprets ``train.num_epochs`` as an
absolute final epoch. Therefore this script reads the latest checkpoint and
sets ``num_epochs = latest_epoch + 1 + additional_epochs``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

import prepare_nf_generalize_fig2_configs as base


CONTINUE_SWEEP_NAME = "nf_generalize_fig2_u256_continue"
DEFAULT_ADDITIONAL_UPDATES = 400_000
CHECKPOINT_EVERY_UPDATES = 5_000
DEFAULT_DATASET_TAGS = ("d2p11", "d2p12", "d2p13", "d2p14", "d2p15")
EPOCH_RE = re.compile(r"checkpoint-epoch-(\d+)$")


def checkpoint_epoch(path: Path) -> int | None:
    match = EPOCH_RE.search(path.name)
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


def ceil_updates_to_epochs(steps_per_epoch: int, updates: int) -> int:
    return max(1, math.ceil(int(updates) / int(steps_per_epoch)))


def checkpoint_epochs_for(steps_per_epoch: int, checkpoint_every_updates: int) -> int:
    return max(1, round(int(checkpoint_every_updates) / int(steps_per_epoch)))


def selected_run_names(args: argparse.Namespace) -> set[str]:
    rows = base.iter_runs()
    selected = {
        row["run_name"]
        for row in rows
        if row["arch"] == "u256"
        and (args.include_all_u256 or row["dataset_tag"] in set(args.dataset_tag or DEFAULT_DATASET_TAGS))
    }
    if args.run_name:
        selected &= set(args.run_name)
    return selected


def continue_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = selected_run_names(args)
    rows: list[dict[str, Any]] = []
    for row in base.iter_runs():
        if row["run_name"] not in selected:
            continue
        dataset_size = int(row["dataset_size"])
        batch_size = int(row["batch_size"])
        steps_per_epoch = base.steps_per_epoch(dataset_size, batch_size)
        latest_epoch = latest_checkpoint_epoch(Path(row["checkpoint_dir"]))
        resume_start_epoch = 0 if latest_epoch is None else latest_epoch + 1
        additional_epochs = ceil_updates_to_epochs(steps_per_epoch, args.additional_updates)
        final_num_epochs = resume_start_epoch + additional_epochs
        additional_actual_updates = additional_epochs * steps_per_epoch
        updated = dict(row)
        updated.update(
            {
                "continue_sweep_name": CONTINUE_SWEEP_NAME,
                "latest_checkpoint_epoch_at_prepare": latest_epoch,
                "resume_start_epoch": int(resume_start_epoch),
                "base_target_updates": int(row["target_updates"]),
                "base_actual_updates": int(row["actual_updates"]),
                "additional_target_updates": int(args.additional_updates),
                "additional_epochs": int(additional_epochs),
                "additional_actual_updates": int(additional_actual_updates),
                "final_num_epochs": int(final_num_epochs),
                "final_nominal_updates": int(final_num_epochs * steps_per_epoch),
                "checkpoint_every_updates": int(args.checkpoint_every_updates),
                "checkpoint_every_n_epochs": int(
                    checkpoint_epochs_for(steps_per_epoch, args.checkpoint_every_updates)
                ),
                "config": f"local/{CONTINUE_SWEEP_NAME}/configs/{row['run_name']}.yaml",
                "checkpoint_dir": row["checkpoint_dir"],
                "note": (
                    "UNet-256 continuation config: reuse the original checkpoint "
                    "directory and add optimizer updates from the latest checkpoint."
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
    config["train"]["num_epochs"] = int(row["final_num_epochs"])
    config["train"]["checkpoint_every_n_epochs"] = int(row["checkpoint_every_n_epochs"])
    return config


def assert_config(path: Path, row: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)
    checks = {
        "io.output_dir": config["io"]["output_dir"] == row["checkpoint_dir"],
        "train.num_epochs": config["train"].get("num_epochs") == row["final_num_epochs"],
        "train.checkpoint_every_n_epochs": (
            config["train"].get("checkpoint_every_n_epochs") == row["checkpoint_every_n_epochs"]
        ),
        "data.no_augmentation": "augmentations" not in config,
        "data.target_size": sum(config["data"]["n_samples"]) * base.SLICES_PER_VOLUME == row["dataset_size"],
        "model.u256": config["model"]["kwargs"].get("block_out_channels") == base.ARCHES["u256"]["block_out_channels"],
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
    parser.add_argument("--dataset-tag", action="append", help="Restrict dataset tag, e.g. d2p13. Repeatable.")
    parser.add_argument("--include-all-u256", action="store_true", help="Include all u256 dataset sizes.")
    parser.add_argument(
        "--additional-updates",
        type=int,
        default=DEFAULT_ADDITIONAL_UPDATES,
        help="Approximate additional optimizer updates to run from the latest checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=CHECKPOINT_EVERY_UPDATES,
        help="Approximate update interval between checkpoints.",
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
            "dataset_tag",
            "dataset_size",
            "steps_per_epoch",
            "latest_checkpoint_epoch_at_prepare",
            "resume_start_epoch",
            "additional_target_updates",
            "additional_actual_updates",
            "additional_epochs",
            "final_num_epochs",
            "checkpoint_every_n_epochs",
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
