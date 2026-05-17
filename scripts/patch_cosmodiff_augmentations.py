#!/usr/bin/env python
"""Patch extra augmentation support into a local ``cosmo_diffusion`` checkout.

This is a small, idempotent patch for the external checkout used on Great
Lakes.  It fixes ``RandomFlip`` so it flips the configured dimensions, then
adds square-image symmetry augmentations used by the focused augmentation
sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXTRA_CLASSES = '''

class RandomRot90(nn.Module):
    """Randomly rotate a tensor by k * 90 degrees along two dimensions.

    Args:
        dims (tuple of int): The two image dimensions to rotate.
        p (float): Probability of applying the augmentation.  Defaults to 1.
    """
    def __init__(self, dims=(-2, -1), p=1.0):
        super().__init__()
        self.dims = tuple(dims)
        if len(self.dims) != 2:
            raise ValueError("RandomRot90 requires exactly two dims.")
        self.p = float(p)

    def __call__(self, x):
        if x is None:
            return None
        if self.p < 1.0 and torch.rand((), device='cpu').item() >= self.p:
            return x
        k = int(torch.randint(0, 4, (1,), device='cpu').item())
        return torch.rot90(x, k, dims=self.dims)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims}, p={self.p})"


class RandomDihedral2D(nn.Module):
    """Randomly apply one of the eight square symmetries.

    This is equivalent to a random 0/90/180/270 degree rotation plus an
    optional reflection along the first configured image dimension.
    """
    def __init__(self, dims=(-2, -1), p=1.0):
        super().__init__()
        self.dims = tuple(dims)
        if len(self.dims) != 2:
            raise ValueError("RandomDihedral2D requires exactly two dims.")
        self.p = float(p)

    def __call__(self, x):
        if x is None:
            return None
        if self.p < 1.0 and torch.rand((), device='cpu').item() >= self.p:
            return x
        k = int(torch.randint(0, 4, (1,), device='cpu').item())
        x = torch.rot90(x, k, dims=self.dims)
        if torch.rand((), device='cpu').item() < 0.5:
            x = torch.flip(x, [self.dims[0]])
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims}, p={self.p})"
'''


def source_for_patch(path: Path) -> tuple[str, Path]:
    backup = path.with_suffix(path.suffix + ".codex_augmentations.bak")
    return path.read_text(), backup


def patch_random_flip(source: str) -> tuple[str, bool]:
    fixed = (
        "        flip_mask = torch.rand(len(self.dims), device='cpu') < self.p\n"
        "        flip_dims = [dim for dim, do_flip in zip(self.dims, flip_mask.tolist()) if do_flip]\n"
        "        return torch.flip(x, flip_dims) if flip_dims else x\n"
    )
    if fixed in source:
        return source, False

    old = (
        "        flip = torch.rand(len(self.dims), device='cpu')\n"
        "        flip = torch.where(flip < self.p)[0].tolist()\n"
        "        return torch.flip(x, flip)\n"
    )
    if old not in source:
        return source, False
    return source.replace(old, fixed, 1), True


def patch_extra_classes(source: str) -> tuple[str, bool]:
    changed = False
    updated = source
    if "class RandomRot90" not in updated or "class RandomDihedral2D" not in updated:
        marker = "\n\nclass RandomMove"
        if marker not in updated:
            marker = "\n\ndef config_augmentations"
        if marker not in updated:
            raise ValueError("Could not find insertion point in cosmodiff/augment.py")
        updated = updated.replace(marker, EXTRA_CLASSES + marker, 1)
        changed = True
    return updated, changed


def patch_augment(path: Path) -> bool:
    source, backup = source_for_patch(path)
    updated, changed_flip = patch_random_flip(source)
    updated, changed_classes = patch_extra_classes(updated)
    changed = changed_flip or changed_classes or updated != path.read_text()
    if changed:
        if not backup.exists():
            backup.write_text(source)
        path.write_text(updated)
    print(
        "cosmo_diffusion augmentation patch:",
        f"random_flip={'patched' if changed_flip else 'ok'}",
        f"extra_classes={'patched' if changed_classes else 'ok'}",
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    augment_path = args.cosmodiff_dir / "cosmodiff" / "augment.py"
    if not augment_path.exists():
        raise FileNotFoundError(f"Missing {augment_path}")
    patch_augment(augment_path)


if __name__ == "__main__":
    main()
