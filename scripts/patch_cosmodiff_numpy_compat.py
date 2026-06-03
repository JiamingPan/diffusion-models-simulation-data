#!/usr/bin/env python
"""Patch cosmodiff NumPy calls for older NumPy versions on Great Lakes."""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "Codex NumPy asarray-copy compatibility patch"


def source_for_patch(path: Path) -> tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".codex_numpy_compat.bak")
    if backup.exists():
        return backup.read_text(), backup
    return path.read_text(), backup


def patch_utils(path: Path) -> bool:
    current = path.read_text()
    if PATCH_MARKER in current:
        print("cosmo_diffusion numpy compatibility patch: ok")
        return False

    source, backup = source_for_patch(path)
    needle = "images = np.asarray(images, copy=True)"
    if needle not in source:
        print("cosmo_diffusion numpy compatibility patch: not-needed")
        return False

    replacement = (
        f"# {PATCH_MARKER}: np.asarray(copy=...) requires newer NumPy.\n"
        "                images = np.array(images, copy=True)"
    )
    updated = source.replace(needle, replacement, 1)
    if not backup.exists():
        backup.write_text(source)
    path.write_text(updated)
    print("cosmo_diffusion numpy compatibility patch: patched")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    utils_path = args.cosmodiff_dir / "cosmodiff" / "utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Missing {utils_path}")
    patch_utils(utils_path)


if __name__ == "__main__":
    main()
