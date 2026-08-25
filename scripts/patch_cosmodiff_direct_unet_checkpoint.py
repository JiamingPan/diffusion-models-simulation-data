#!/usr/bin/env python
"""Patch cosmodiff checkpoint resume to avoid diffusers.AutoModel.

Recent diffusers versions import optional pipeline-loading dependencies from
``AutoModel.from_pretrained``.  The Great Lakes checkpoints used here are either
plain/class-conditional ``UNet2DModel`` checkpoints or continuous-conditional
``UNet2DConditionModel`` checkpoints, so loading the concrete class directly is
both clearer and avoids optional packages such as httpx.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "Codex direct checkpoint model-class patch v3"


def source_for_patch(path: Path) -> tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".codex_direct_unet_checkpoint.bak")
    if backup.exists():
        return backup.read_text(), backup
    return path.read_text(), backup


def patch_utils(path: Path) -> bool:
    source, backup = source_for_patch(path)
    if PATCH_MARKER in path.read_text():
        print("cosmo_diffusion direct checkpoint model-class patch: ok")
        return False

    needle = "model = AutoModel.from_pretrained(ckpt_path)"
    if needle not in source:
        print("cosmo_diffusion direct checkpoint model-class patch: not-needed")
        return False

    indent = ""
    for line in source.splitlines():
        if needle in line:
            indent = line[: len(line) - len(line.lstrip())]
            break

    replacement = "\n".join(
        [
            f"{indent}# {PATCH_MARKER}: avoid AutoModel optional pipeline imports.",
            f"{indent}from simdiff_eval.torch_compat import install_torch_backend_compat",
            f"{indent}install_torch_backend_compat(entry_point='patched_cosmodiff.utils.load_checkpoint')",
            f"{indent}import json as _codex_json",
            f"{indent}with open(os.path.join(ckpt_path, \"config.json\")) as _codex_f:",
            f"{indent}    _codex_model_config = _codex_json.load(_codex_f)",
            f"{indent}_codex_class = _codex_model_config.get(\"_class_name\") or _codex_model_config.get(\"class_name\")",
            f"{indent}_codex_blocks = list(_codex_model_config.get(\"down_block_types\", [])) + list(_codex_model_config.get(\"up_block_types\", []))",
            f"{indent}if _codex_class == \"UNet2DConditionModel\" or any(\"CrossAttn\" in str(_codex_block) for _codex_block in _codex_blocks):",
            f"{indent}    from diffusers import UNet2DConditionModel",
            f"{indent}    model = UNet2DConditionModel.from_pretrained(ckpt_path)",
            f"{indent}else:",
            f"{indent}    try:",
            f"{indent}        from diffusers import UNet2DModel",
            f"{indent}        model = UNet2DModel.from_pretrained(ckpt_path)",
            f"{indent}    except Exception:",
            f"{indent}        model = AutoModel.from_pretrained(ckpt_path)",
        ]
    )
    updated = source.replace(f"{indent}{needle}", replacement, 1)
    if not backup.exists():
        backup.write_text(source)
    path.write_text(updated)
    print("cosmo_diffusion direct checkpoint model-class patch: patched")
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
