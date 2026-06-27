#!/usr/bin/env python
"""Patch older cosmo_diffusion data loading for constant labels.

The unconditional DiT Fig.2 sweep uses ``data.constant_label: 0`` so every
image gets the same null class label.  Recent ``cosmo_diffusion`` supports this
in ``utils.parse_config_data``.  Older Great Lakes checkouts ignore the key,
which makes DiT receive ``class_labels=None`` and crash inside diffusers adaLN.

This patch inserts the small constant-label block before ``ArrayDataset`` is
constructed.  It is idempotent and keeps a backup of the original file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "codex constant-label patch"


def source_for_patch(path: Path) -> str:
    backup = path.with_suffix(path.suffix + ".codex_constant_label.bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    return path.read_text()


def patch_utils(path: Path) -> bool:
    source = source_for_patch(path)
    if MARKER in source or "constant_label = data_cfg.get(\"constant_label\", None)" in source:
        print("cosmo_diffusion constant-label patch: ok")
        return False

    return_pattern = re.compile(
        r"^(?P<indent>\s*)return ArrayDataset\(images, labels=labels(?P<tail>[^\n]*)\)",
        flags=re.MULTILINE,
    )
    match = return_pattern.search(source)
    if match is not None:
        indent = match.group("indent")
        tail = match.group("tail")
        replacement = (
            f"{indent}# {MARKER}: make unconditional DiT runs provide a null class label.\n"
            f"{indent}constant_label = data_cfg.get(\"constant_label\", None)\n"
            f"{indent}if labels is None and constant_label is not None:\n"
            f"{indent}    labels = torch.full(\n"
            f"{indent}        (len(images),),\n"
            f"{indent}        int(constant_label),\n"
            f"{indent}        dtype=torch.long,\n"
            f"{indent}        device=images.device,\n"
            f"{indent}    )\n"
            f"\n"
            f"{indent}return ArrayDataset(images, labels=labels{tail})"
        )

        path.write_text(return_pattern.sub(replacement, source, count=1))
        print("cosmo_diffusion constant-label patch: patched")
        return True

    dict_pattern = re.compile(
        r"^(?P<indent>[ \t]*)output = \{\s*$(?=\n[ \t]*['\"]data['\"]:\s*ArrayDataset\()",
        flags=re.MULTILINE,
    )
    match = dict_pattern.search(source)
    if match is None:
        raise RuntimeError(
            f"Could not find the expected ArrayDataset construction in {path}; "
            "inspect cosmodiff.utils.parse_config_data."
        )

    indent = match.group("indent")
    replacement = (
        f"{indent}# {MARKER}: make unconditional DiT runs provide a null class label.\n"
        f"{indent}constant_label = data_cfg.get(\"constant_label\", None)\n"
        f"{indent}if out.get(\"labels\") is None and constant_label is not None:\n"
        f"{indent}    out[\"labels\"] = torch.full(\n"
        f"{indent}        (len(out[\"images\"]),),\n"
        f"{indent}        int(constant_label),\n"
        f"{indent}        dtype=torch.long,\n"
        f"{indent}        device=out[\"images\"].device,\n"
        f"{indent}    )\n"
        f"\n"
        f"{indent}output = {{"
    )

    path.write_text(dict_pattern.sub(replacement, source, count=1))
    print("cosmo_diffusion constant-label patch: patched")
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
