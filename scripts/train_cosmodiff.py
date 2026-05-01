#!/usr/bin/env python
"""Thin training wrapper around nkern/cosmo_diffusion's cosmodiff_train.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--cosmodiff-train",
        default="cosmo_diffusion/scripts/cosmodiff_train.py",
        help="Path to cosmodiff_train.py from nkern/cosmo_diffusion.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Optional extra args passed after '--' to cosmodiff_train.py.",
    )
    args = parser.parse_args()

    train_script = Path(args.cosmodiff_train)
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find training script: {train_script}")

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    cmd = [sys.executable, str(train_script), "--config", args.config, *extra_args]
    print("Running:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
