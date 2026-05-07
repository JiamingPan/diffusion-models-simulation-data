#!/usr/bin/env python
"""Compute a paper-style SSCD generalizability curve.

For each generated sample ``x``, this compares the SSCD descriptor to all real
training slices ``y_i`` used by that run:

    GL = 1 - P(max_i sim_sscd(x, y_i) > threshold)

This is a near-copy/generalization diagnostic. It is intentionally separate
from P(k), which is a physics-fidelity diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/templates/reproducibility_manifest_template.json")
    parser.add_argument("--arch", default="u64", help="Architecture to plot, e.g. u64, u128, u256.")
    parser.add_argument("--sample-root", default="results/tables/samples")
    parser.add_argument("--config-dir", required=True, help="Directory containing per-run cosmodiff YAML configs.")
    parser.add_argument(
        "--sscd-path",
        default=os.environ.get("SSCD_PATH", "/home/jiamingp/.cache/torch/hub/sscd_disc_mixup.torchscript.pt"),
        help="Path to sscd_disc_mixup.torchscript.pt. Can also be set via SSCD_PATH.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=32, help="SSCD embedding batch size.")
    parser.add_argument("--similarity-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=320, help="Square SSCD input size.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit for auto.")
    parser.add_argument(
        "--render-mode",
        choices=("fixed", "per_image"),
        default="fixed",
        help="fixed preserves training-space amplitudes; per_image emphasizes morphology.",
    )
    parser.add_argument(
        "--max-real",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. Omit for the full paper-style comparison.",
    )
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--output-csv", default="results/tables/generalizability_sscd.csv")
    parser.add_argument("--output-figure", default="results/figures/generalizability_sscd.png")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a JSON list.")
    return rows


def row_dataset_size(row: dict[str, Any]) -> float:
    for key in ("dataset_size", "actual_2d", "target_2d"):
        if key in row and row[key] is not None:
            return float(row[key])
    raise ValueError(f"Manifest row {row.get('run_name')} has no dataset-size field.")


def sample_path(row: dict[str, Any], sample_root: Path, manifest_dir: Path, seed: int) -> Path:
    if row.get("sample_path"):
        raw = str(row["sample_path"]).format(seed=seed, run_name=row["run_name"])
        path = Path(raw)
        return path if path.is_absolute() else manifest_dir / path
    return sample_root / f"{row['run_name']}_seed{seed}.npy"


def maybe_limit(arr: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or len(arr) <= limit:
        return arr
    # Deterministic spread through the array, not a random subset.
    idx = np.linspace(0, len(arr) - 1, limit, dtype=int)
    return arr[idx]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: float(row["dataset_size"]))

    x = np.array([float(row["dataset_size"]) for row in rows])
    y = np.array([float(row["generalization_score"]) for row in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o", lw=2.5, ms=7, color="steelblue")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("dataset size")
    ax.set_ylabel("SSCD generalization score")
    ax.set_title(title)
    ax.grid(alpha=0.25)

    def xfmt(value: float, _pos: int) -> str:
        if value <= 0:
            return ""
        exponent = int(round(np.log2(value)))
        if np.isclose(value, 2**exponent):
            return rf"$2^{{{exponent}}}$"
        return f"{value:g}"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(xfmt))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    cosmodiff_candidate = project_root / "cosmo_diffusion"
    if cosmodiff_candidate.exists() and str(cosmodiff_candidate) not in sys.path:
        sys.path.insert(0, str(cosmodiff_candidate))

    from simdiff_eval.io import as_nchw, load_npy, load_real_from_config
    from simdiff_eval.sscd import load_sscd_torchscript, sscd_embeddings, sscd_generalization_metrics

    manifest_path = Path(args.manifest)
    sample_root = Path(args.sample_root)
    config_dir = Path(args.config_dir)
    rows = [row for row in load_manifest(manifest_path) if row["arch"] == args.arch]
    rows = sorted(rows, key=row_dataset_size)

    if not rows:
        raise ValueError(f"No rows found for arch={args.arch!r}.")

    model = load_sscd_torchscript(args.sscd_path, device=args.device)
    device = args.device or next(model.parameters()).device

    records: list[dict[str, Any]] = []
    for row in rows:
        run_name = row["run_name"]
        generated_path = sample_path(row, sample_root, manifest_path.parent, args.seed)
        config_path = config_dir / f"{run_name}.yaml"

        if not generated_path.exists() or not config_path.exists():
            message = f"Missing input for {run_name}: sample={generated_path.exists()} config={config_path.exists()}"
            if args.skip_missing:
                print("SKIP", message)
                continue
            raise FileNotFoundError(message)

        generated = as_nchw(load_npy(generated_path))
        generated = maybe_limit(generated, args.max_generated)
        real_training = load_real_from_config(config_path)
        real_training = maybe_limit(real_training, args.max_real)

        print(
            f"{run_name}: generated={len(generated)} real_training={len(real_training)} "
            f"dataset_size={row_dataset_size(row):.0f}"
        )
        gen_emb = sscd_embeddings(
            generated,
            model,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            render_mode=args.render_mode,
        )
        real_emb = sscd_embeddings(
            real_training,
            model,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            render_mode=args.render_mode,
        )
        metrics = sscd_generalization_metrics(
            gen_emb,
            real_emb,
            threshold=args.threshold,
            batch_size=args.similarity_batch_size,
        )

        record = {
            "run_name": run_name,
            "arch": args.arch,
            "dataset_tag": row["dataset_tag"],
            "dataset_size": row_dataset_size(row),
            "n_generated": len(generated),
            "n_real_training_compared": len(real_training),
            "render_mode": args.render_mode,
            **metrics,
        }
        records.append(record)
        print(
            f"  GL={record['generalization_score']:.3f} "
            f"copy_fraction={record['copy_fraction']:.3f} "
            f"max_sim_median={record['max_similarity_median']:.3f}"
        )

    if not records:
        raise RuntimeError("No records were computed.")

    write_csv(records, Path(args.output_csv))
    plot(records, Path(args.output_figure), f"SSCD generalizability ({args.arch})")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_figure}")


if __name__ == "__main__":
    main()
