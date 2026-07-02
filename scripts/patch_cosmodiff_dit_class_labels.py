#!/usr/bin/env python
"""Patch older cosmo_diffusion training loops for DiT class labels.

``diffusers.DiTTransformer2DModel`` uses adaptive layer norm conditioning and
requires ``class_labels`` in ``forward()``.  The Fig.2 DiT sweep is still
unconditional, so every image receives the same null label from
``data.constant_label: 0``.  Older external ``cosmo_diffusion`` checkouts load
that label but do not pass it into the model call inside ``optim.train``.

This patch is intentionally small and idempotent so Slurm jobs can apply it to
the external checkout before training.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "codex DiT class-label patch"


def source_for_patch(path: Path) -> str:
    backup = path.with_suffix(path.suffix + ".codex_dit_class_labels.bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    return path.read_text()


def patch_optim(path: Path) -> bool:
    source = source_for_patch(path)
    if MARKER in source:
        print("cosmo_diffusion DiT class-label patch: ok")
        return False

    pattern = re.compile(
        r"^(?P<indent>\s*)pred = model\(noisy_images, timesteps, return_dict=False\)\[0\]",
        flags=re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        if "class_labels=labels" in source or "class_labels=batch_labels" in source:
            print("cosmo_diffusion DiT class-label patch: already supported")
            return False
        raise RuntimeError(
            f"Could not find the expected unconditioned model call in {path}; inspect cosmodiff.optim.train."
        )

    indent = match.group("indent")
    replacement = (
        f"{indent}# {MARKER}: pass dataset labels to DiT adaLN when present.\n"
        f"{indent}batch_labels = None\n"
        f"{indent}if isinstance(batch, dict):\n"
        f"{indent}    batch_labels = batch.get(\"labels\")\n"
        f"{indent}elif isinstance(batch, (tuple, list)) and len(batch) >= 2:\n"
        f"{indent}    batch_labels = batch[1]\n"
        f"{indent}if batch_labels is not None:\n"
        f"{indent}    if torch.is_tensor(batch_labels):\n"
        f"{indent}        batch_labels = batch_labels.to(device=noisy_images.device, dtype=torch.long)\n"
        f"{indent}    else:\n"
        f"{indent}        batch_labels = torch.as_tensor(batch_labels, device=noisy_images.device, dtype=torch.long)\n"
        f"{indent}    pred = model(\n"
        f"{indent}        noisy_images,\n"
        f"{indent}        timestep=timesteps,\n"
        f"{indent}        class_labels=batch_labels,\n"
        f"{indent}        return_dict=False,\n"
        f"{indent}    )[0]\n"
        f"{indent}else:\n"
        f"{indent}    pred = model(noisy_images, timesteps, return_dict=False)[0]"
    )

    path.write_text(pattern.sub(replacement, source, count=1))
    print("cosmo_diffusion DiT class-label patch: patched")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    optim_path = args.cosmodiff_dir / "cosmodiff" / "optim.py"
    if not optim_path.exists():
        raise FileNotFoundError(f"Missing {optim_path}")
    patch_optim(optim_path)


if __name__ == "__main__":
    main()
