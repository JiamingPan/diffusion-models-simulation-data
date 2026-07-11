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
CHECKPOINT_MARKER = "codex DiT checkpoint-class patch"


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


def patch_checkpoint_loader(path: Path) -> bool:
    """Make checkpoint resume reconstruct the saved diffusers model class."""
    source = source_for_patch(path)
    if CHECKPOINT_MARKER in source:
        print("cosmo_diffusion DiT checkpoint-class patch: ok")
        return False

    patterns = (
        re.compile(
            r"^(?P<indent>\s*)model = AutoModel\.from_pretrained\(ckpt_path\)",
            flags=re.MULTILINE,
        ),
        re.compile(
            r"^(?P<indent>\s*)model = diffusers\.UNet2DModel\.from_pretrained\(ckpt_path\)",
            flags=re.MULTILINE,
        ),
    )
    match = next((pattern.search(source) for pattern in patterns if pattern.search(source)), None)
    if match is None:
        if 'model_config.get("_class_name")' in source and "getattr(diffusers, class_name)" in source:
            print("cosmo_diffusion DiT checkpoint-class patch: already supported")
            return False
        raise RuntimeError(
            f"Could not find the expected checkpoint model loader in {path}; "
            "inspect cosmodiff.utils.load_checkpoint."
        )

    indent = match.group("indent")
    replacement = (
        f"{indent}# {CHECKPOINT_MARKER}: reconstruct the class recorded by save_pretrained.\n"
        f"{indent}model_config_path = os.path.join(ckpt_path, \"config.json\")\n"
        f"{indent}if not os.path.exists(model_config_path):\n"
        f"{indent}    raise FileNotFoundError(f\"Missing diffusers config: {{model_config_path}}\")\n"
        f"{indent}with open(model_config_path) as model_config_file:\n"
        f"{indent}    model_config = json.load(model_config_file)\n"
        f"{indent}class_name = model_config.get(\"_class_name\")\n"
        f"{indent}if not class_name:\n"
        f"{indent}    raise ValueError(f\"Checkpoint {{ckpt_path!r}} does not record _class_name\")\n"
        f"{indent}try:\n"
        f"{indent}    model_cls = getattr(diffusers, class_name)\n"
        f"{indent}except AttributeError as exc:\n"
        f"{indent}    raise ValueError(\n"
        f"{indent}        f\"Checkpoint {{ckpt_path!r}} requires unavailable diffusers class {{class_name!r}}\"\n"
        f"{indent}    ) from exc\n"
        f"{indent}model = model_cls.from_pretrained(ckpt_path)\n"
        f"{indent}meta_parameters = [\n"
        f"{indent}    name for name, parameter in model.named_parameters()\n"
        f"{indent}    if parameter.device.type == \"meta\"\n"
        f"{indent}]\n"
        f"{indent}if meta_parameters:\n"
        f"{indent}    raise RuntimeError(\n"
        f"{indent}        f\"Checkpoint {{ckpt_path!r}} left meta parameters after loading: {{meta_parameters[:8]}}\"\n"
        f"{indent}    )"
    )

    patched = source[: match.start()] + replacement + source[match.end() :]
    path.write_text(patched)
    print("cosmo_diffusion DiT checkpoint-class patch: patched")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    optim_path = args.cosmodiff_dir / "cosmodiff" / "optim.py"
    utils_path = args.cosmodiff_dir / "cosmodiff" / "utils.py"
    if not optim_path.exists():
        raise FileNotFoundError(f"Missing {optim_path}")
    if not utils_path.exists():
        raise FileNotFoundError(f"Missing {utils_path}")
    patch_optim(optim_path)
    patch_checkpoint_loader(utils_path)


if __name__ == "__main__":
    main()
