#!/usr/bin/env python
"""Audit the frozen fresh DiT-L16 400k manifest before Slurm submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_TAGS = [f"d2p{power:02d}" for power in range(6, 16)]
EXPECTED_STAGES = list(range(1, 17))
SCIENTIFIC_UPDATES = [200_000, 300_000, 400_000]


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return rows


def audit(rows: list[dict], *, require_empty: bool) -> None:
    if len(rows) != 160:
        raise ValueError(f"Expected 160 stage rows, found {len(rows)}")
    if sorted({row["dataset_tag"] for row in rows}) != EXPECTED_TAGS:
        raise ValueError("Manifest does not contain exactly d2p06 through d2p15")
    if sorted({int(row["stage"]) for row in rows}) != EXPECTED_STAGES:
        raise ValueError("Manifest does not contain exactly stages 1 through 16")
    if {int(row["target_total_updates"]) for row in rows if row["stage"] == 16} != {
        400_000
    }:
        raise ValueError("Every final-stage run must target 400000 updates")
    if sorted(
        {
            int(row["target_total_updates"])
            for row in rows
            if row["scientific_checkpoint"]
        }
    ) != SCIENTIFIC_UPDATES:
        raise ValueError("Scientific milestones do not match the frozen three-checkpoint plan")

    checkpoint_dirs = {Path(row["checkpoint_dir"]) for row in rows}
    if len(checkpoint_dirs) != 10:
        raise ValueError(f"Expected ten isolated checkpoint directories, found {len(checkpoint_dirs)}")
    for row in rows:
        if not row["fresh_initialization"] or int(row["training_seed"]) != 123:
            raise ValueError("Every run must be a seed-123 fresh initialization")
        if "fresh400k" not in row["run_name"]:
            raise ValueError(f"Non-fresh run name in manifest: {row['run_name']}")
        if "nf_generalize_fig2_dit_l16_continue" in row["checkpoint_dir"]:
            raise ValueError(f"Legacy continuation path in manifest: {row['checkpoint_dir']}")
        if "fresh300k" in row["checkpoint_dir"]:
            raise ValueError(f"Superseded fresh300k path in manifest: {row['checkpoint_dir']}")

    if require_empty:
        nonempty = []
        for checkpoint_dir in sorted(checkpoint_dirs):
            if any(checkpoint_dir.glob("checkpoint-epoch-*")):
                nonempty.append(str(checkpoint_dir))
        if nonempty:
            raise ValueError(
                "Fresh submission requires empty checkpoint directories:\n"
                + "\n".join(nonempty)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--require-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit(load_rows(args.manifest), require_empty=args.require_empty)
    mode = "empty fresh start" if args.require_empty else "restart-safe"
    print(f"Fresh DiT-L16 400k manifest audit PASS ({mode}).")


if __name__ == "__main__":
    main()
