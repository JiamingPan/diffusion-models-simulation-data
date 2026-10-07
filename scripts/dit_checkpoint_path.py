"""Resolve only the planned epoch, accepting native zero-padded names."""
from pathlib import Path


def resolve_checkpoint(root, epoch):
    root = Path(root)
    epoch = int(epoch)
    matches = [p for p in root.glob('checkpoint-epoch-*')
               if p.is_dir() and p.name.removeprefix('checkpoint-epoch-').isdigit()
               and int(p.name.removeprefix('checkpoint-epoch-')) == epoch]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one checkpoint for epoch {epoch} in {root}; found {matches}')
    return matches[0]
