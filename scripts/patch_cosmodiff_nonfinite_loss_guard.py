#!/usr/bin/env python
"""Patch cosmo_diffusion training to stop immediately on non-finite loss."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "Non-finite training loss"
OLD = """            batch_loss = loss.detach().item()

            global_step += 1
"""
NEW = """            batch_loss = loss.detach().item()
            if not np.isfinite(batch_loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, step {global_step}: {batch_loss}"
                )

            global_step += 1
"""


def patch_optim(path: Path) -> bool:
    backup = path.with_suffix(path.suffix + ".codex_nonfinite_loss_guard.bak")
    source = path.read_text()
    if MARKER in source:
        print("cosmo_diffusion non-finite-loss guard: ok")
        return False
    if OLD not in source:
        raise RuntimeError(f"Could not find training batch_loss block in {path}")
    if not backup.exists():
        backup.write_text(source)
    path.write_text(source.replace(OLD, NEW, 1))
    print("cosmo_diffusion non-finite-loss guard: patched")
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
