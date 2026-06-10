#!/usr/bin/env python
"""Plot generated samples from several training checkpoints.

The companion Slurm script writes one ``.npz`` file per selected checkpoint
epoch.  This script loads those files and makes a compact grid for a poster or
talk: rows are training epochs and columns are generated samples.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_sample_array(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in ("samples", "images", "arr_0"):
            if key in data:
                arr = data[key]
                break
        else:
            arr = data[data.files[0]]
    else:
        arr = data

    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[1] in (1, 3):
            arr = arr[:, 0]
        elif arr.shape[-1] in (1, 3):
            arr = arr[..., 0]
        else:
            raise ValueError(f"Cannot infer image channel axis for {path}: shape={arr.shape}")
    if arr.ndim != 3:
        raise ValueError(f"Expected samples with shape (N,H,W), got {arr.shape} from {path}")
    return arr


def _epoch_from_name(path: Path) -> int:
    match = re.search(r"_epoch(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not parse epoch from {path.name}")
    return int(match.group(1))


def _discover_files(sample_dir: Path, run_name: str, sample_label: str) -> list[Path]:
    pattern = f"{run_name}_epoch*_seed*_{sample_label}.npz"
    files = sorted(sample_dir.glob(pattern), key=_epoch_from_name)
    if not files:
        raise FileNotFoundError(f"No files matched {sample_dir / pattern}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sample-label", default="dpm50")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster-output", type=Path, default=None)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--title", default="Generated samples during training")
    args = parser.parse_args()

    files = _discover_files(args.sample_dir, args.run_name, args.sample_label)
    rows = []
    for path in files:
        arr = _load_sample_array(path)[: args.max_samples]
        rows.append((_epoch_from_name(path), arr))

    n_rows = len(rows)
    n_cols = max(arr.shape[0] for _, arr in rows)
    chosen = np.concatenate([arr[:n_cols] for _, arr in rows], axis=0)
    vmin, vmax = np.percentile(chosen, [1.0, 99.0])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 16,
            "axes.titlesize": 18,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.15 * n_cols + 1.4, 2.15 * n_rows + 0.9),
        squeeze=False,
    )

    for r, (epoch, arr) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < arr.shape[0]:
                ax.imshow(arr[c], cmap=args.cmap, vmin=vmin, vmax=vmax)
            else:
                ax.axis("off")
            if r == 0:
                ax.set_title(f"sample {c + 1}", pad=6)
            if c == 0:
                ax.set_ylabel(f"epoch {epoch}", rotation=0, ha="right", va="center", labelpad=48)

    fig.suptitle(args.title, y=0.995, fontsize=22)
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.02, top=0.91, wspace=0.04, hspace=0.14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if args.poster_output is not None:
        args.poster_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.poster_output, bbox_inches="tight")
    print(f"wrote {args.output}")
    if args.poster_output is not None:
        print(f"wrote {args.poster_output}")


if __name__ == "__main__":
    main()
