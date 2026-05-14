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
import re
from pathlib import Path


SAFE_STEP_MARKER = "step_call_kwargs"


def infer_model_output_name(context: str) -> str:
    """Infer the denoiser prediction variable used before scheduler.step."""
    for name in ("noise_pred", "model_output", "model_pred", "pred", "output"):
        if re.search(rf"^\s*{re.escape(name)}\s*=", context, flags=re.MULTILINE):
            return name
    if '"model_output": noise_pred' in context:
        return "model_output"
    return "model_output"


def replace_scheduler_step_call(source: str) -> tuple[str, bool]:
    if SAFE_STEP_MARKER in source:
        return source, False

    lines = source.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0

    while i < len(lines):
        line = lines[i]
        window = "".join(lines[i : min(i + 16, len(lines))])
        is_prior_bad_patch = (
            "step_kwargs = {" in line
            and '"model_output":' in window
            and "images = noise_scheduler.step(**step_kwargs).prev_sample" in window
        )
        is_original_step_call = "images = noise_scheduler.step(" in line
        if not is_original_step_call and not is_prior_bad_patch:
            out.append(line)
            i += 1
            continue

        indent = line[: len(line) - len(line.lstrip())]
        model_output_name = infer_model_output_name("".join(out[-80:]))
        paren_balance = 0
        if is_prior_bad_patch:
            while i < len(lines):
                if "images = noise_scheduler.step(**step_kwargs).prev_sample" in lines[i]:
                    i += 1
                    break
                i += 1
        else:
            while i < len(lines):
                paren_balance += lines[i].count("(") - lines[i].count(")")
                if ".prev_sample" in lines[i] and paren_balance <= 0:
                    i += 1
                    break
                i += 1
        out.extend(
            [
                f"{indent}step_call_kwargs = {{}}\n",
                f"{indent}if \"step_kwargs\" in locals():\n",
                f"{indent}    step_call_kwargs.update(step_kwargs)\n",
                f"{indent}if \"generator\" in noise_scheduler.step.__code__.co_varnames:\n",
                f"{indent}    step_call_kwargs[\"generator\"] = generator\n",
                f"{indent}images = noise_scheduler.step({model_output_name}, t, images, **step_call_kwargs).prev_sample\n",
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


def source_for_patch(path: Path) -> tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".codex_sampling_compat.bak")
    if backup.exists():
        return backup.read_text(), backup
    return path.read_text(), backup


def patch_optim(path: Path) -> bool:
    source, backup = source_for_patch(path)
    updated, changed_step = replace_scheduler_step_call(source)
    updated, changed_ema = skip_duplicate_ema_symlinks(updated)
    changed = changed_step or changed_ema or updated != path.read_text()
    if changed:
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
    source, backup = source_for_patch(path)
    updated, changed_unwrap = unwrap_synthesized_ema_model(source)
    changed = changed_unwrap or updated != path.read_text()
    if changed:
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
