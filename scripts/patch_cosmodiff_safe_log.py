#!/usr/bin/env python
"""Patch cosmo_diffusion's log transform to avoid non-finite zero voxels.

The upstream ``Transform(log=True)`` path calls ``x.log_()`` directly.  That is
fine for strictly positive single-field runs, but multi-field CAMELS runs can
contain exact zeros.  A single zero becomes ``-inf`` and then normalization can
turn the entire fitted tensor into ``nan``.

This patch keeps the existing ``transform: ["log"]`` config semantics for
positive data while flooring zeros to a tiny positive value before logging.
It is intentionally idempotent so Slurm jobs can apply it at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "COSMODIFF_SAFE_LOG_FLOOR"
OLD = """        if self.log:
            x.log_()
"""
NEW = """        if self.log:
            # COSMODIFF_SAFE_LOG_FLOOR: CAMELS multi-field grids can contain exact zeros.
            x.clamp_min_(1.0e-12).log_()
"""


def patch_transform(path: Path) -> bool:
    backup = path.with_suffix(path.suffix + ".codex_safe_log.bak")
    source = path.read_text()
    if MARKER in source:
        print("cosmo_diffusion safe-log patch: ok")
        return False
    if OLD not in source:
        raise RuntimeError(f"Could not find upstream log transform block in {path}")
    if not backup.exists():
        backup.write_text(source)
    path.write_text(source.replace(OLD, NEW, 1))
    print("cosmo_diffusion safe-log patch: patched")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    transform_path = args.cosmodiff_dir / "cosmodiff" / "transform.py"
    if not transform_path.exists():
        raise FileNotFoundError(f"Missing {transform_path}")
    patch_transform(transform_path)


if __name__ == "__main__":
    main()
