#!/usr/bin/env python
"""Compare DDPM500 and DPM-Solver50 samples for the class-conditional run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import prepare_nf_class_conditional_u128_config as base


SAMPLERS = ("ddpm500", "dpm50")


def load_npz_samples(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "samples" in data:
            arr = data["samples"]
        else:
            arr = data[data.files[0]]
    return np.asarray(arr, dtype=np.float32)


def default_sample_path(project_dir: Path, run_name: str, seed: int, sampler: str, sample_n: int) -> Path:
    return (
        project_dir
        / "results"
        / base.SWEEP_NAME
        / "sampler_compare"
        / f"{run_name}_seed{seed}_{sampler}_n{sample_n}.npz"
    )


def summarize(arr: np.ndarray) -> dict[str, float]:
    flat = arr.reshape(arr.shape[0], -1)
    return {
        "pixel_mean": float(flat.mean()),
        "pixel_std": float(flat.std()),
        "pixel_p01": float(np.percentile(flat, 1)),
        "pixel_p50": float(np.percentile(flat, 50)),
        "pixel_p99": float(np.percentile(flat, 99)),
        "sample_mean_std": float(flat.mean(axis=1).std()),
        "sample_std_mean": float(flat.std(axis=1).mean()),
    }


def write_stats(
    *,
    output_dir: Path,
    run_name: str,
    labels: np.ndarray,
    class_map: dict[str, int],
    samples: dict[str, np.ndarray],
) -> None:
    rows: list[dict[str, Any]] = []
    by_sampler_class: dict[tuple[str, str], dict[str, float]] = {}
    for sampler, arr in samples.items():
        rows.append({"sampler": sampler, "field": "all", "class_id": "all", **summarize(arr)})
        by_sampler_class[(sampler, "all")] = summarize(arr)
        for field, class_id in class_map.items():
            sub = arr[labels == class_id]
            stats = summarize(sub)
            rows.append({"sampler": sampler, "field": field, "class_id": class_id, **stats})
            by_sampler_class[(sampler, field)] = stats

    compare_rows = []
    for field in ["all", *class_map.keys()]:
        ddpm = by_sampler_class[("ddpm500", field)]
        dpm = by_sampler_class[("dpm50", field)]
        compare_rows.append(
            {
                "field": field,
                "abs_mean_delta": abs(dpm["pixel_mean"] - ddpm["pixel_mean"]),
                "abs_std_delta": abs(dpm["pixel_std"] - ddpm["pixel_std"]),
                "abs_p01_delta": abs(dpm["pixel_p01"] - ddpm["pixel_p01"]),
                "abs_p99_delta": abs(dpm["pixel_p99"] - ddpm["pixel_p99"]),
            }
        )

    stats_path = output_dir / f"{run_name}_sampler_compare_stats.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    deltas_path = output_dir / f"{run_name}_sampler_compare_deltas.csv"
    with deltas_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    summary_path = output_dir / f"{run_name}_sampler_compare_summary.json"
    summary = {
        "run_name": run_name,
        "samplers": list(samples),
        "n_samples": int(labels.size),
        "class_counts": {
            field: int(np.sum(labels == class_id))
            for field, class_id in class_map.items()
        },
        "delta_csv": str(deltas_path),
        "stats_csv": str(stats_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {stats_path}")
    print(f"Wrote {deltas_path}")
    print(f"Wrote {summary_path}")


def write_histogram_plot(
    *,
    output_dir: Path,
    run_name: str,
    labels: np.ndarray,
    class_map: dict[str, int],
    samples: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    axes = axes.ravel()
    bins = np.linspace(
        min(float(arr.min()) for arr in samples.values()),
        max(float(arr.max()) for arr in samples.values()),
        80,
    )
    for ax, (field, class_id) in zip(axes, class_map.items()):
        for sampler, arr in samples.items():
            sub = arr[labels == class_id].reshape(-1)
            ax.hist(sub, bins=bins, histtype="step", density=True, linewidth=1.5, label=sampler)
        ax.set_title(f"{class_id}: {field}")
        ax.set_xlabel("pixel value")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Class-conditional sampler comparison: DDPM500 vs DPM-Solver50")
    path = output_dir / f"{run_name}_sampler_compare_histograms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample-n", type=int, default=96)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    run_name = args.run_name or base.run_name()
    output_dir = Path(args.output_dir) if args.output_dir else project_dir / "results" / base.SWEEP_NAME / "sampler_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = (
        project_dir
        / "local"
        / base.SWEEP_NAME
        / "labels"
        / f"{run_name}_sample_class_labels_n{args.sample_n}.npy"
    )
    class_map_path = (
        project_dir
        / "local"
        / base.SWEEP_NAME
        / "labels"
        / f"{run_name}_class_map.json"
    )
    labels = np.load(labels_path)
    class_map = json.loads(class_map_path.read_text())
    samples = {
        sampler: load_npz_samples(default_sample_path(project_dir, run_name, args.seed, sampler, args.sample_n))
        for sampler in SAMPLERS
    }
    for sampler, arr in samples.items():
        if arr.shape[0] != labels.size:
            raise ValueError(f"{sampler} sample count {arr.shape[0]} != labels {labels.size}")
        if arr.ndim != 4 or arr.shape[1:] != (1, 128, 128):
            raise ValueError(f"{sampler} expected shape (N, 1, 128, 128), got {arr.shape}")

    write_stats(
        output_dir=output_dir,
        run_name=run_name,
        labels=labels,
        class_map=class_map,
        samples=samples,
    )
    if not args.skip_plot:
        write_histogram_plot(
            output_dir=output_dir,
            run_name=run_name,
            labels=labels,
            class_map=class_map,
            samples=samples,
        )


if __name__ == "__main__":
    main()
