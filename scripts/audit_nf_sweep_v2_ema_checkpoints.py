#!/usr/bin/env python
"""Audit nf_sweep_v2 checkpoint and post-hoc EMA snapshot consistency.

This script is read-only.  It is meant to catch cases where raw sampling and
post-hoc EMA synthesis may not be comparing cleanly, especially after resumed
training jobs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CHECKPOINT_ROOT = Path(
    "/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_sweep_v2"
)
DEFAULT_MANIFEST = Path("local/nf_sweep_v2/manifest.json")


def checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"checkpoint-epoch-(\d+)$", path.name)
    return int(match.group(1)) if match else None


def ema_file_info(path: Path) -> tuple[int | None, int | None]:
    """Return ``(profile_index, step)`` parsed from ``0.12345.pt`` names."""
    parts = path.name.split(".")
    if len(parts) < 3 or parts[-1] != "pt":
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def selected_rows(manifest: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = manifest
    if args.run_name:
        keep = set(args.run_name)
        rows = [row for row in rows if row["run_name"] in keep]
    if args.arch:
        keep = set(args.arch)
        rows = [row for row in rows if row["arch"] in keep]
    if args.variant:
        keep = set(args.variant)
        rows = [row for row in rows if row["variant_tag"] in keep]
    return rows


def audit_run(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    run_name = row["run_name"]
    ckpt_root = args.checkpoint_root / f"{run_name}_checkpoints"
    config_path = Path(row["config"])

    result: dict[str, Any] = {
        "run_name": run_name,
        "arch": row.get("arch"),
        "variant": row.get("variant_tag"),
        "checkpoint_root": str(ckpt_root),
        "latest_epoch": None,
        "expected_final_epoch": None,
        "n_checkpoints": 0,
        "n_ema_files": 0,
        "duplicate_ema_names": 0,
        "profiles": "",
        "flags": [],
    }

    if config_path.exists():
        cfg = read_yaml(config_path)
        num_epochs = int(cfg.get("train", {}).get("num_epochs", 0))
        if num_epochs > 0:
            result["expected_final_epoch"] = num_epochs - 1
    else:
        result["flags"].append("MISSING_CONFIG")

    ckpts = sorted(
        [p for p in ckpt_root.glob("checkpoint-epoch-*") if p.is_dir()],
        key=lambda p: checkpoint_epoch(p) if checkpoint_epoch(p) is not None else -1,
    )
    result["n_checkpoints"] = len(ckpts)
    if not ckpts:
        result["flags"].append("NO_CHECKPOINTS")
        return result

    epochs = [checkpoint_epoch(p) for p in ckpts]
    latest_epoch = max(e for e in epochs if e is not None)
    result["latest_epoch"] = latest_epoch
    expected = result["expected_final_epoch"]
    if expected is not None and latest_epoch != expected:
        result["flags"].append("LATEST_EPOCH_NE_CONFIG")

    ema_files_by_name: dict[str, list[Path]] = defaultdict(list)
    profile_steps: dict[int, list[int]] = defaultdict(list)
    missing_ema_dirs = 0
    for ckpt in ckpts:
        ema_dir = ckpt / "ema"
        if not ema_dir.exists():
            missing_ema_dirs += 1
            continue
        for path in sorted(ema_dir.glob("*.pt")):
            ema_files_by_name[path.name].append(path)
            profile, step = ema_file_info(path)
            if profile is not None and step is not None:
                profile_steps[profile].append(step)

    result["n_ema_files"] = sum(len(v) for v in ema_files_by_name.values())
    duplicates = {name: paths for name, paths in ema_files_by_name.items() if len(paths) > 1}
    result["duplicate_ema_names"] = len(duplicates)
    if missing_ema_dirs:
        result["flags"].append(f"MISSING_EMA_DIRS={missing_ema_dirs}")
    if not ema_files_by_name:
        result["flags"].append("NO_EMA_FILES")
    if duplicates:
        result["flags"].append("DUPLICATE_EMA_FILENAMES")

    profile_summary = []
    for profile, steps in sorted(profile_steps.items()):
        counts = Counter(steps)
        dup_steps = sum(1 for count in counts.values() if count > 1)
        profile_summary.append(
            f"{profile}:n={len(steps)},min={min(steps)},max={max(steps)},dup_steps={dup_steps}"
        )
    result["profiles"] = ";".join(profile_summary)

    if args.details and duplicates:
        result["duplicate_examples"] = {
            name: [str(p) for p in paths[:5]]
            for name, paths in list(sorted(duplicates.items()))[: args.max_duplicate_examples]
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--arch", action="append", choices=["u64", "u128"])
    parser.add_argument("--variant", action="append")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.add_argument("--max-duplicate-examples", type=int, default=5)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows = selected_rows(manifest, args)
    results = [audit_run(row, args) for row in rows]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    columns = [
        "run_name",
        "latest_epoch",
        "expected_final_epoch",
        "n_checkpoints",
        "n_ema_files",
        "duplicate_ema_names",
        "profiles",
        "flags",
    ]
    print("\t".join(columns))
    for row in results:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, list):
                value = ",".join(value) if value else "OK"
            values.append(str(value))
        print("\t".join(values))

    bad = [row for row in results if row["flags"]]
    print(f"\nAudited {len(results)} runs; flagged {len(bad)} runs.")
    if bad:
        print("Most important flags:")
        for row in bad[:10]:
            print(f"- {row['run_name']}: {', '.join(row['flags'])}")


if __name__ == "__main__":
    main()
