#!/usr/bin/env python
"""Audit the clean DiT-L16 300k manifest before Slurm submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_TAGS = [f"d2p{power:02d}" for power in range(6, 16)]


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return rows


def audit(rows: list[dict], *, require_empty: bool) -> None:
    if len(rows) != 10:
        raise ValueError(f"Expected ten run rows, found {len(rows)}")
    if sorted(row["dataset_tag"] for row in rows) != EXPECTED_TAGS:
        raise ValueError("Manifest does not contain exactly d2p06 through d2p15")
    if {int(row["target_total_updates"]) for row in rows} != {300_000}:
        raise ValueError("Every run must target 300000 updates")

    checkpoint_dirs = {Path(row["checkpoint_dir"]) for row in rows}
    if len(checkpoint_dirs) != 10:
        raise ValueError("Every data size must have an isolated checkpoint directory")
    for row in rows:
        if not row["fresh_initialization"] or int(row["training_seed"]) != 123:
            raise ValueError("Every run must be a seed-123 fresh initialization")
        if "fresh300k_v2" not in row["run_name"]:
            raise ValueError(f"Unexpected run identity: {row['run_name']}")
        if "fresh400k" in row["checkpoint_dir"] or "l16_continue" in row["checkpoint_dir"]:
            raise ValueError(f"Old checkpoint path in clean manifest: {row['checkpoint_dir']}")

    if require_empty:
        nonempty = [
            str(path)
            for path in sorted(checkpoint_dirs)
            if any(path.glob("checkpoint-epoch-*"))
        ]
        if nonempty:
            raise ValueError(
                "Clean submission requires empty checkpoint directories:\n"
                + "\n".join(nonempty)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--require-empty", action="store_true")
    args = parser.parse_args()
    audit(load_rows(args.manifest), require_empty=args.require_empty)
    mode = "empty clean start" if args.require_empty else "restart-safe"
    print(f"Fresh DiT-L16 300k v2 manifest audit PASS ({mode}).")


if __name__ == "__main__":
    main()
