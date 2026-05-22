#!/usr/bin/env python
"""Patch cosmo_diffusion label loading for continuous conditioning.

Some cosmodiff versions cast every label array to ``torch.long`` inside
``utils.load_data``.  That is correct for class labels, but it destroys
normalized CAMELS parameter vectors before they reach ``encoder_hidden_states``.

This patch keeps integer label arrays as ``long`` and floating label arrays as
``float32``.  It is intentionally small and idempotent so Slurm jobs can apply
it to the external cosmo_diffusion checkout before training.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "label_dtype = torch.float32 if torch.is_floating_point(label_tensor) else torch.long"


def source_for_patch(path: Path) -> tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".codex_conditional_labels.bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    return path.read_text(), backup


def patch_utils(path: Path) -> bool:
    source, _backup = source_for_patch(path)
    if MARKER in source:
        print("cosmo_diffusion continuous-label patch: ok")
        return False

    patterns = [
        re.compile(
            r"^(?P<indent>\s*)labels = torch\.as_tensor\(labels, device=device, dtype=torch\.long\)",
            flags=re.MULTILINE,
        ),
        re.compile(
            r"^(?P<indent>\s*)labels = torch\.as_tensor\(labels, dtype=torch\.long\)",
            flags=re.MULTILINE,
        ),
    ]

    updated = source
    changed = False
    for pattern in patterns:
        if pattern.search(updated):
            updated = pattern.sub(
                lambda match: (
                    f"{match.group('indent')}label_tensor = torch.as_tensor(labels)\n"
                    f"{match.group('indent')}{MARKER}\n"
                    f"{match.group('indent')}labels = torch.as_tensor(labels, device=device, dtype=label_dtype)"
                ),
                updated,
            )
            changed = True

    if not changed:
        raise RuntimeError(
            f"Could not find the expected label cast in {path}; inspect cosmodiff.utils.load_data."
        )

    path.write_text(updated)
    print("cosmo_diffusion continuous-label patch: patched")
    return True


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
