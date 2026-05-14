#!/usr/bin/env python
"""Patch known ``cosmo_diffusion`` sampling compatibility issues.

This is intentionally small and idempotent.  It patches the external
``cosmo_diffusion`` checkout used on Great Lakes until the upstream fixes land.

Fixes:
- Some diffusers schedulers, including ``HeunDiscreteScheduler``, do not accept
  ``generator=`` in ``scheduler.step(...)``.
- Post-hoc EMA synthesis can see duplicated EMA checkpoint filenames when EMA
  folders are copied into multiple training checkpoints.
- Some ``cosmodiff_sample.py`` versions keep the synthesized ``KarrasEMA``
  wrapper instead of unwrapping its ``ema_model`` before reading ``model.config``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SAFE_STEP_MARKER = '"model_output": noise_pred'


def replace_scheduler_step_call(source: str) -> tuple[str, bool]:
    if SAFE_STEP_MARKER in source and "noise_scheduler.step(**step_kwargs)" in source:
        return source, False

    lines = source.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0

    while i < len(lines):
        line = lines[i]
        if "images = noise_scheduler.step(" not in line:
            out.append(line)
            i += 1
            continue

        indent = line[: len(line) - len(line.lstrip())]
        paren_balance = 0
        while i < len(lines):
            paren_balance += lines[i].count("(") - lines[i].count(")")
            if ".prev_sample" in lines[i] and paren_balance <= 0:
                i += 1
                break
            i += 1
        out.extend(
            [
                f"{indent}step_kwargs = {{\n",
                f"{indent}    \"model_output\": noise_pred,\n",
                f"{indent}    \"timestep\": t,\n",
                f"{indent}    \"sample\": images,\n",
                f"{indent}}}\n",
                f"{indent}if \"generator\" in noise_scheduler.step.__code__.co_varnames:\n",
                f"{indent}    step_kwargs[\"generator\"] = generator\n",
                f"{indent}images = noise_scheduler.step(**step_kwargs).prev_sample\n",
            ]
        )
        changed = True
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


def unwrap_synthesized_ema_model(source: str) -> tuple[str, bool]:
    marker = "if hasattr(model, \"ema_model\"):"
    if marker in source:
        return source, False
    if "model = synthesize_ema_from_checkpoints(" not in source:
        return source, False

    lines = source.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if "model = synthesize_ema_from_checkpoints(" not in line:
            i += 1
            continue

        indent = line[: len(line) - len(line.lstrip())]
        paren_balance = line.count("(") - line.count(")")
        i += 1
        while i < len(lines):
            out.append(lines[i])
            paren_balance += lines[i].count("(") - lines[i].count(")")
            i += 1
            if paren_balance <= 0:
                out.extend(
                    [
                        f"{indent}if hasattr(model, \"ema_model\"):\n",
                        f"{indent}    model = model.ema_model\n",
                    ]
                )
                changed = True
                break

    return "".join(out), changed


def patch_optim(path: Path) -> bool:
    source = path.read_text()
    updated, changed_step = replace_scheduler_step_call(source)
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


def patch_sampler_script(path: Path) -> bool:
    source = path.read_text()
    updated, changed_unwrap = unwrap_synthesized_ema_model(source)
    if changed_unwrap:
        backup = path.with_suffix(path.suffix + ".codex_sampling_compat.bak")
        if not backup.exists():
            backup.write_text(source)
        path.write_text(updated)
    print(
        "cosmo_diffusion sampler script patch:",
        f"ema_unwrap={'patched' if changed_unwrap else 'ok'}",
    )
    return changed_unwrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    optim_path = args.cosmodiff_dir / "cosmodiff" / "optim.py"
    if not optim_path.exists():
        raise FileNotFoundError(f"Missing {optim_path}")
    patch_optim(optim_path)

    sampler_path = args.cosmodiff_dir / "scripts" / "cosmodiff_sample.py"
    if not sampler_path.exists():
        raise FileNotFoundError(f"Missing {sampler_path}")
    patch_sampler_script(sampler_path)


if __name__ == "__main__":
    main()
