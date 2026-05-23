#!/usr/bin/env python
"""Patch cosmo_diffusion for diffusers versions without ``AutoModel``.

Great Lakes currently runs this path with torch 1.12.1, which rules out recent
diffusers releases.  diffusers 0.31.0 can run the class-conditional UNet, but
some cosmo_diffusion checkouts import ``diffusers.AutoModel`` unconditionally.
This patch makes that import optional and falls back to ``UNet2DModel`` for
checkpoint loading, which is the model class used by the u128 class run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "class _CosmodiffAutoModelFallback:"
ORIGINAL_IMPORT = "from diffusers import AutoModel, DDPMScheduler"
PATCHED_IMPORT = """\
try:
    from diffusers import AutoModel, DDPMScheduler
except ImportError:
    from diffusers import DDPMScheduler, UNet2DModel

    class _CosmodiffAutoModelFallback:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return UNet2DModel.from_pretrained(*args, **kwargs)

    AutoModel = _CosmodiffAutoModelFallback
"""


def source_for_patch(path: Path) -> tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".codex_diffusers_automodel.bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    return path.read_text(), backup


def patch_file(path: Path) -> bool:
    source, _backup = source_for_patch(path)
    if MARKER in source:
        print(f"cosmo_diffusion AutoModel patch: ok {path}")
        return False
    if ORIGINAL_IMPORT not in source:
        print(f"cosmo_diffusion AutoModel patch: skipped {path}")
        return False

    path.write_text(source.replace(ORIGINAL_IMPORT, PATCHED_IMPORT, 1))
    print(f"cosmo_diffusion AutoModel patch: patched {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    changed = False
    for rel_path in ("cosmodiff/utils.py", "cosmodiff/optim.py"):
        path = args.cosmodiff_dir / rel_path
        if path.exists():
            changed = patch_file(path) or changed
    if not changed:
        print("cosmo_diffusion AutoModel patch: no changes needed")


if __name__ == "__main__":
    main()
