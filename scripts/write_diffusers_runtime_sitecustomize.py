#!/usr/bin/env python
"""Write a thin adapter for the canonical Torch compatibility module."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.seed_restart_runtime import write_sitecustomize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to write, usually results/cache/python_stubs/sitecustomize.py.")
    parser.add_argument("--code-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--entry-point",
        default="generated.sitecustomize",
        help="Entry-point name recorded by the canonical compatibility marker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = write_sitecustomize(
        Path(args.path),
        code_root=args.code_root,
        entry_point=args.entry_point,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
