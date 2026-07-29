#!/usr/bin/env python
"""Patch cosmodiff checkpoints so strict training resume has complete state."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


MARKER = "codex complete checkpoint state"
REQUIRED_FILENAMES = (
    "optimizer.pkl",
    "noise_scheduler.pkl",
    "lr_scheduler.pkl",
)
SAVE_MODEL_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)accelerator\.unwrap_model\(model\)"
    r"\.save_pretrained\(ckpt_save_path\)[ \t]*$",
    flags=re.MULTILINE,
)


def _ensure_import(source: str, module: str) -> str:
    if re.search(rf"^(?:import|from)[ \t]+{re.escape(module)}\b", source, re.MULTILINE):
        return source
    lines = source.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    while insert_at < len(lines) and (
        lines[insert_at].startswith("#")
        or not lines[insert_at].strip()
        or lines[insert_at].startswith("from __future__")
    ):
        insert_at += 1
    lines.insert(insert_at, f"import {module}\n")
    return "".join(lines)


def patch_checkpoint_state(optim_path: Path) -> bool:
    """Add explicit optimizer, diffusion scheduler, and LR scheduler pickles."""
    optim_path = optim_path.expanduser().resolve()
    source = optim_path.read_text()
    present = [name for name in REQUIRED_FILENAMES if name in source]
    if len(present) == len(REQUIRED_FILENAMES):
        return False
    if present:
        raise RuntimeError(
            f"{optim_path} contains only part of the strict-resume state "
            f"({', '.join(present)}); inspect it before patching."
        )

    match = SAVE_MODEL_PATTERN.search(source)
    if match is None:
        raise RuntimeError(
            "Could not find the model checkpoint save call in "
            f"{optim_path}; inspect the cosmodiff checkpoint block."
        )

    indent = match.group("indent")
    insertion = (
        f"\n{indent}# {MARKER}\n"
        f'{indent}checkpoint_optimizer = getattr(optimizer, "optimizer", optimizer)\n'
        f'{indent}checkpoint_lr_scheduler = getattr(lr_scheduler, "scheduler", lr_scheduler)\n'
        f'{indent}with open(os.path.join(ckpt_save_path, "optimizer.pkl"), "wb") as handle:\n'
        f"{indent}    pickle.dump(checkpoint_optimizer, handle)\n"
        f'{indent}with open(os.path.join(ckpt_save_path, "noise_scheduler.pkl"), "wb") as handle:\n'
        f"{indent}    pickle.dump(noise_scheduler, handle)\n"
        f'{indent}with open(os.path.join(ckpt_save_path, "lr_scheduler.pkl"), "wb") as handle:\n'
        f"{indent}    pickle.dump(checkpoint_lr_scheduler, handle)"
    )
    patched = source[: match.end()] + insertion + source[match.end() :]
    patched = _ensure_import(patched, "os")
    patched = _ensure_import(patched, "pickle")
    compile(patched, str(optim_path), "exec")

    backup = optim_path.with_suffix(optim_path.suffix + ".codex_checkpoint_state.bak")
    if not backup.exists():
        shutil.copy2(optim_path, backup)
    optim_path.write_text(patched)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    optim_path = args.cosmodiff_dir.expanduser().resolve() / "cosmodiff" / "optim.py"
    if not optim_path.is_file():
        raise FileNotFoundError(f"Missing cosmodiff optimizer module: {optim_path}")
    changed = patch_checkpoint_state(optim_path)
    print(f"cosmodiff complete-checkpoint patch: {'patched' if changed else 'ok'}")


if __name__ == "__main__":
    main()
