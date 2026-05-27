#!/usr/bin/env python
"""Patch cosmodiff checkpoint resume to avoid diffusers.AutoModel for UNets.

Recent diffusers versions import optional pipeline-loading dependencies from
``AutoModel.from_pretrained``.  Great Lakes training checkpoints here are
``UNet2DModel`` checkpoints, so resuming through AutoModel is unnecessary and
can fail when optional packages such as httpx are absent from the runtime env.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "Codex direct-UNet checkpoint resume patch"


def source_for_patch(path: Path) -> tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".codex_direct_unet_checkpoint.bak")
    if backup.exists():
        return backup.read_text(), backup
    return path.read_text(), backup


def patch_utils(path: Path) -> bool:
    source, backup = source_for_patch(path)
    if PATCH_MARKER in path.read_text():
        print("cosmo_diffusion direct-UNet checkpoint patch: ok")
        return False

    needle = "model = AutoModel.from_pretrained(ckpt_path)"
    if needle not in source:
        print("cosmo_diffusion direct-UNet checkpoint patch: not-needed")
        return False

    indent = ""
    for line in source.splitlines():
        if needle in line:
            indent = line[: len(line) - len(line.lstrip())]
            break

    replacement = "\n".join(
        [
            f"{indent}# {PATCH_MARKER}: all current CAMELS diffusion checkpoints are UNet2DModel.",
            f"{indent}try:",
            f"{indent}    from diffusers import UNet2DModel",
            f"{indent}    model = UNet2DModel.from_pretrained(ckpt_path)",
            f"{indent}except Exception:",
            f"{indent}    model = AutoModel.from_pretrained(ckpt_path)",
        ]
    )
    updated = source.replace(f"{indent}{needle}", replacement, 1)
    if not backup.exists():
        backup.write_text(source)
    path.write_text(updated)
    print("cosmo_diffusion direct-UNet checkpoint patch: patched")
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
