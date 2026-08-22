#!/usr/bin/env python
"""Keep only the newest complete cosmodiff recovery checkpoints."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmodiff_train_with_dit_resume import (
    checkpoint_epoch,
    checkpoint_is_complete,
)


def prune(checkpoint_dir: Path, *, keep: int) -> list[Path]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if keep < 1:
        raise ValueError("keep must be at least one")
    complete = sorted(
        (
            (path, epoch)
            for path in checkpoint_dir.glob("checkpoint-epoch-*")
            if path.is_dir()
            and (epoch := checkpoint_epoch(path)) is not None
            and checkpoint_is_complete(path)
        ),
        key=lambda item: item[1],
    )
    removed = []
    for path, _epoch in complete[:-keep]:
        shutil.rmtree(path)
        removed.append(path)
        print(f"Pruned old complete recovery checkpoint: {path}", flush=True)
    return removed


def quarantine_incomplete(checkpoint_dir: Path) -> list[Path]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    quarantine_dir = checkpoint_dir / "_incomplete_checkpoints"
    moved = []
    for path in sorted(checkpoint_dir.glob("checkpoint-epoch-*")):
        if not path.is_dir() or checkpoint_epoch(path) is None:
            continue
        if checkpoint_is_complete(path):
            continue
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / path.name
        suffix = 1
        while destination.exists():
            destination = quarantine_dir / f"{path.name}.{suffix}"
            suffix += 1
        path.rename(destination)
        moved.append(destination)
        print(f"Quarantined incomplete checkpoint: {path} -> {destination}", flush=True)
    return moved


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--quarantine-incomplete", action="store_true")
    args = parser.parse_args()

    if args.quarantine_incomplete:
        quarantine_incomplete(args.checkpoint_dir)
    if args.watch_pid is None:
        prune(args.checkpoint_dir, keep=args.keep)
        return
    if args.interval_seconds <= 0:
        raise ValueError("interval-seconds must be positive")

    while process_exists(args.watch_pid):
        prune(args.checkpoint_dir, keep=args.keep)
        time.sleep(args.interval_seconds)
    prune(args.checkpoint_dir, keep=args.keep)


if __name__ == "__main__":
    main()
