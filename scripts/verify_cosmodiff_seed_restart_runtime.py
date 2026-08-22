#!/usr/bin/env python
"""Read-only source contract for the pinned external cosmodiff trainer."""

from __future__ import annotations

import argparse
from pathlib import Path


def require_fragments(path: Path, fragments: tuple[str, ...]) -> None:
    source = path.read_text()
    missing = [fragment for fragment in fragments if fragment not in source]
    if missing:
        raise RuntimeError(f"{path} lacks required runtime contracts: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()
    optim = args.cosmodiff_dir / "cosmodiff/optim.py"
    utils = args.cosmodiff_dir / "cosmodiff/utils.py"
    require_fragments(
        optim,
        (
            "resume_from_checkpoint",
            "accelerator.load_state(resume_from_checkpoint)",
            "from ema_pytorch import PostHocEMA",
            "global_step = 0",
            "global_step >= ema_burn_in",
            "ema.checkpoint_folder = Path(ckpt_save_path) / 'ema'",
            "ema.checkpoint()",
            "class_labels",
            "noise_scheduler.save_pretrained(ckpt_save_path)",
            '"checkpoint_config.yaml"',
            "accelerator.save_state(ckpt_save_path)",
        ),
    )
    require_fragments(
        utils,
        (
            "def parse_config_data(config: dict)",
            "class ArrayDataset",
            "self.labels = labels",
        ),
    )
    print(f"PASS: pinned cosmodiff seed-restart source contract at {args.cosmodiff_dir}")


if __name__ == "__main__":
    main()
