#!/usr/bin/env python
"""Patch known ``cosmo_diffusion`` sampling compatibility issues.

This is intentionally small and idempotent.  It patches the external
``cosmo_diffusion`` checkout used on Great Lakes until the upstream fixes land.

Fixes:
- Some diffusers schedulers, including ``HeunDiscreteScheduler``, do not accept
  ``generator=`` in ``scheduler.step(...)``.
- Post-hoc EMA synthesis can see duplicated EMA checkpoint filenames when EMA
  folders are copied into multiple training checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def remove_generator_from_scheduler_step(source: str) -> tuple[str, bool]:
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    in_step = False
    paren_balance = 0

    for line in lines:
        if "noise_scheduler.step(" in line:
            in_step = True
            paren_balance = 0

        if in_step:
            paren_balance += line.count("(") - line.count(")")
            if "generator=generator" in line:
                changed = True
                continue
            out.append(line)
            if paren_balance <= 0:
                in_step = False
            continue

        out.append(line)

    return "".join(out), changed


def skip_duplicate_ema_symlinks(source: str) -> tuple[str, bool]:
    needle = "(tmp_path / pt_file.name).symlink_to(pt_file.resolve())"
    if needle not in source:
        return source, False

    lines = source.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    for line in lines:
        if needle not in line:
            out.append(line)
            continue

        indent = line[: len(line) - len(line.lstrip())]
        out.extend(
            [
                f"{indent}link_path = tmp_path / pt_file.name\n",
                f"{indent}if link_path.exists():\n",
                f"{indent}    continue\n",
                f"{indent}link_path.symlink_to(pt_file.resolve())\n",
            ]
        )
        changed = True

    return "".join(out), changed


def patch_file(path: Path) -> bool:
    source = path.read_text()
    updated, changed_step = remove_generator_from_scheduler_step(source)
    updated, changed_ema = skip_duplicate_ema_symlinks(updated)
    changed = changed_step or changed_ema
    if changed:
        backup = path.with_suffix(path.suffix + ".codex_sampling_compat.bak")
        if not backup.exists():
            backup.write_text(source)
        path.write_text(updated)
    print(
        "cosmo_diffusion sampling compatibility patch:",
        f"scheduler_step={'patched' if changed_step else 'ok'}",
        f"ema_symlinks={'patched' if changed_ema else 'ok'}",
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    optim_path = args.cosmodiff_dir / "cosmodiff" / "optim.py"
    if not optim_path.exists():
        raise FileNotFoundError(f"Missing {optim_path}")
    patch_file(optim_path)


if __name__ == "__main__":
    main()
