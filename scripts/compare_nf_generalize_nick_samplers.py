#!/usr/bin/env python
"""Compare DPM-Solver50 Nick-data samples against existing DDPM500 samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from compute_nf_generalize_pca_full_nn import (
    DEFAULT_SAMPLE_LABEL,
    load_manifest,
    load_npz_array,
    sample_path_for,
    selected_rows,
)


def evenly_limit(arr: np.ndarray, limit: int | None) -> np.ndarray:
    arr = np.asarray(arr)
    if limit is None or len(arr) <= limit:
        return np.array(arr, copy=True)
    idx = np.linspace(0, len(arr) - 1, limit, dtype=int)
    return np.array(arr[idx], copy=True)


def summarize(arr: np.ndarray) -> dict[str, float]:
    flat = arr.reshape(arr.shape[0], -1)
    return {
        "pixel_mean": float(np.mean(flat)),
        "pixel_std": float(np.std(flat)),
        "pixel_p01": float(np.percentile(flat, 1)),
        "pixel_p50": float(np.percentile(flat, 50)),
        "pixel_p99": float(np.percentile(flat, 99)),
        "sample_mean_std": float(np.std(np.mean(flat, axis=1))),
        "sample_std_mean": float(np.mean(np.std(flat, axis=1))),
    }


def histogram_l1(a: np.ndarray, b: np.ndarray, bins: int = 120) -> float:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a.reshape(-1), bins=edges, density=True)
    hb, _ = np.histogram(b.reshape(-1), bins=edges, density=True)
    width = float(np.mean(np.diff(edges)))
    return float(np.sum(np.abs(ha - hb)) * width)


def write_plot(rows: list[dict[str, Any]], output_dir: Path, dpm_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["dataset_size"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(x, [row["ddpm_pixel_std"] for row in rows], "o-", label="DDPM500")
    axes[0].plot(x, [row["dpm_pixel_std"] for row in rows], "o-", label="DPM-Solver50")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("training 2D slices")
    axes[0].set_ylabel("pixel std")
    axes[0].legend(frameon=False)

    axes[1].plot(x, [row["hist_l1"] for row in rows], "o-", color="tab:purple")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("training 2D slices")
    axes[1].set_ylabel("DDPM vs DPM histogram L1")
    fig.suptitle(f"Nick-default sampler smoke comparison: raw_train_full vs {dpm_label}")
    path = output_dir / f"nf_generalize_nick_data_{dpm_label}_sampler_compare.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--arch", action="append")
    parser.add_argument("--dataset-tag", action="append")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ddpm-label", default=DEFAULT_SAMPLE_LABEL)
    parser.add_argument("--dpm-label", default="dpm50")
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-missing-samples", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    output_dir = args.output_dir or project_dir / "results" / "nf_generalize_nick_data" / "sampler_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = selected_rows(load_manifest(project_dir, args.manifest), args)
    if not rows:
        raise SystemExit("No rows selected.")

    metric_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        ddpm_path = sample_path_for(project_dir, row, args.seed, args.ddpm_label)
        dpm_path = sample_path_for(project_dir, row, args.seed, args.dpm_label)
        if not ddpm_path.exists() or not dpm_path.exists():
            message = f"{row['run_name']}: ddpm={ddpm_path.exists()} dpm={dpm_path.exists()}"
            if args.skip_missing_samples:
                print(f"skipping missing sample: {message}")
                continue
            missing.append(message)
            continue

        ddpm = evenly_limit(load_npz_array(ddpm_path), args.max_samples)
        dpm = evenly_limit(load_npz_array(dpm_path), args.max_samples)
        n = min(len(ddpm), len(dpm))
        ddpm = ddpm[:n]
        dpm = dpm[:n]
        ddpm_stats = summarize(ddpm)
        dpm_stats = summarize(dpm)
        metric_rows.append(
            {
                "run_name": row["run_name"],
                "dataset_tag": row["dataset_tag"],
                "dataset_size": int(row["dataset_size"]),
                "n_compared": int(n),
                "ddpm_sample_path": str(ddpm_path),
                "dpm_sample_path": str(dpm_path),
                "hist_l1": histogram_l1(ddpm, dpm),
                "abs_mean_delta": abs(dpm_stats["pixel_mean"] - ddpm_stats["pixel_mean"]),
                "abs_std_delta": abs(dpm_stats["pixel_std"] - ddpm_stats["pixel_std"]),
                "abs_p01_delta": abs(dpm_stats["pixel_p01"] - ddpm_stats["pixel_p01"]),
                "abs_p99_delta": abs(dpm_stats["pixel_p99"] - ddpm_stats["pixel_p99"]),
                **{f"ddpm_{k}": v for k, v in ddpm_stats.items()},
                **{f"dpm_{k}": v for k, v in dpm_stats.items()},
            }
        )

    if missing:
        raise FileNotFoundError("Missing sampler comparison inputs:\n" + "\n".join(missing))
    if not metric_rows:
        raise SystemExit("No comparison rows available.")

    metric_rows.sort(key=lambda row: row["dataset_size"])
    csv_path = output_dir / f"nf_generalize_nick_data_{args.dpm_label}_sampler_compare.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_path = output_dir / f"nf_generalize_nick_data_{args.dpm_label}_sampler_compare_summary.json"
    summary = {
        "ddpm_label": args.ddpm_label,
        "dpm_label": args.dpm_label,
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "n_rows": len(metric_rows),
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    if not args.skip_plot:
        write_plot(metric_rows, output_dir, args.dpm_label)


if __name__ == "__main__":
    main()
