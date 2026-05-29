#!/usr/bin/env python
"""Patch cosmo_diffusion MultiNormalization label collection.

Older cosmodiff ``load_data`` builds the class list for multipath
normalization with ``torch.cat([lbls[0] for lbls in labels])``.  Each
``lbls[0]`` is a scalar tensor, so ``torch.cat`` fails with
``zero-dimensional tensor`` as soon as a config uses per-field normalization.

The intended code is the same as the shared-normalization branch:
``torch.cat([lbls[:1] for lbls in labels])``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


BAD = "unq_labels = torch.cat([lbls[0] for lbls in labels])"
GOOD = "unq_labels = torch.cat([lbls[:1] for lbls in labels])"


def source_for_patch(path: Path) -> tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".codex_multinorm_labels.bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    return path.read_text(), backup


def patch_utils(path: Path) -> bool:
    source, _backup = source_for_patch(path)
    if BAD in source:
        path.write_text(source.replace(BAD, GOOD))
        print("cosmo_diffusion multinorm-label patch: patched")
        return True
    if GOOD not in source:
        raise RuntimeError(
            f"Could not find the expected MultiNormalization label line in {path}; inspect cosmodiff.utils.load_data."
        )
    print("cosmo_diffusion multinorm-label patch: ok")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    utils_path = args.cosmodiff_dir / "cosmodiff" / "utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Missing {utils_path}")
    patch_utils(utils_path)


if __name__ == "__main__":
    main()
