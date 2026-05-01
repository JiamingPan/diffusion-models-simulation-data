#!/usr/bin/env python
"""Make a Figure-1-style reproducibility plot from generated sample sets.

This script compares generated samples from different model widths at the same
training-set size. The score is a transparent project diagnostic:

    error = P(k) log10 MAE + |mean_a - mean_b| + |std_a - std_b|
    score = 1 / (1 + error)

It is intended for CAMELS diffusion experiments, but it does not claim to be
the exact metric from any paper unless that paper's exact score is implemented
separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PAIRS = ("u64:u128", "u64:u256", "u128:u256")
DEFAULT_COLORS = {
    "u64_vs_u128": "red",
    "u64_vs_u256": "blue",
    "u128_vs_u256": "limegreen",
}
DEFAULT_MARKERS = {
    "u64_vs_u128": "o",
    "u64_vs_u256": "s",
    "u128_vs_u256": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "JSON manifest listing generated sample sets. See "
            "configs/templates/reproducibility_manifest_template.json."
        ),
    )
    parser.add_argument(
        "--sample-root",
        default="results/tables/samples",
        help="Directory used for sample paths when a manifest row has no sample_path.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed suffix used in default sample filenames.")
    parser.add_argument("--output-csv", default="results/tables/reproducibility_scores.csv")
    parser.add_argument("--output-figure", default="results/figures/reproducibility_scores.png")
    parser.add_argument("--nbins-pk", type=int, default=25)
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="Architecture pair as ARCH_A:ARCH_B. Can be passed multiple times.",
    )
    parser.add_argument(
        "--x-field",
        default="auto",
        help="Manifest field used as the x-axis dataset size. Use auto to prefer dataset_size, actual_2d, then target_2d.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing sample files instead of failing.",
    )
    parser.add_argument(
        "--memorization-max",
        type=float,
        default=None,
        help="Optional x-value where the memorization-regime shading ends.",
    )
    parser.add_argument(
        "--generalization-min",
        type=float,
        default=None,
        help="Optional x-value where the generalization-regime shading starts.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a JSON list of run rows.")

    required = {"run_name", "arch", "dataset_tag"}
    for i, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Manifest row {i} is missing required keys: {sorted(missing)}")
    return rows


def parse_pairs(pair_args: list[str] | None) -> list[tuple[str, str]]:
    raw_pairs = pair_args or list(DEFAULT_PAIRS)
    pairs = []
    for raw in raw_pairs:
        if ":" not in raw:
            raise ValueError(f"Pair must look like ARCH_A:ARCH_B, got {raw!r}.")
        left, right = raw.split(":", 1)
        pairs.append((left, right))
    return pairs


def sample_path(row: dict[str, Any], sample_root: Path, manifest_dir: Path, seed: int) -> Path:
    """Resolve one generated-sample path from a manifest row."""
    if row.get("sample_path"):
        raw = str(row["sample_path"]).format(seed=seed, run_name=row["run_name"])
        path = Path(raw)
        return path if path.is_absolute() else manifest_dir / path

    filename = f"{row['run_name']}_seed{seed}.npy"
    return sample_root / filename


def score_from_pair_metrics(pair_metrics: dict[str, float | str]) -> tuple[float, float]:
    error = (
        float(pair_metrics["pk_log10_mae_between_sets"])
        + float(pair_metrics["mean_abs_mean_diff"])
        + float(pair_metrics["std_abs_diff"])
    )
    return error, 1.0 / (1.0 + error)


def row_dataset_size(row: dict[str, Any], x_field: str) -> float:
    if x_field != "auto":
        if x_field not in row:
            raise ValueError(f"Manifest row {row['run_name']!r} is missing x-field {x_field!r}.")
        return float(row[x_field])

    for key in ("dataset_size", "actual_2d", "target_2d"):
        if key in row:
            return float(row[key])
    raise ValueError(
        f"Manifest row {row['run_name']!r} needs one of dataset_size, actual_2d, or target_2d."
    )


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(
    rows: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    path: Path,
    memorization_max: float | None = None,
    generalization_min: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    x_values = np.array([float(r["dataset_size"]) for r in rows], dtype=float)
    xmin = max(1.0, float(np.nanmin(x_values)) / 1.5)
    xmax = float(np.nanmax(x_values)) * 1.5

    if memorization_max is not None:
        ax.axvspan(xmin, memorization_max, color="tab:red", alpha=0.14, lw=0)
        ax.axvline(memorization_max, color="tab:red", ls="--", lw=2, alpha=0.7)
        ax.text(np.sqrt(xmin * memorization_max), 0.08, "Memorization\nregime", ha="center")

    if generalization_min is not None:
        ax.axvspan(generalization_min, xmax, color="tab:orange", alpha=0.14, lw=0)
        ax.axvline(generalization_min, color="tab:orange", ls="--", lw=2, alpha=0.7)
        ax.text(np.sqrt(generalization_min * xmax), 0.08, "Generalization\nregime", ha="center")

    for pair in pairs:
        pair_key = f"{pair[0]}_vs_{pair[1]}"
        sub = [r for r in rows if r["pair"] == pair_key]
        if not sub:
            continue

        x = np.array([float(r["dataset_size"]) for r in sub], dtype=float)
        y = np.array([float(r["reproducibility_score"]) for r in sub], dtype=float)
        order = np.argsort(x)
        label = f"UNet-{pair[0].lstrip('u')} vs UNet-{pair[1].lstrip('u')}"
        ax.plot(
            x[order],
            y[order],
            color=DEFAULT_COLORS.get(pair_key),
            marker=DEFAULT_MARKERS.get(pair_key, "o"),
            lw=2.5,
            label=label,
        )

    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("dataset size")
    ax.set_ylabel("reproducibility score")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from simdiff_eval.io import as_nchw, load_npy
    from simdiff_eval.metrics import reproducibility_summary

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    sample_root = Path(args.sample_root)
    pairs = parse_pairs(args.pair)

    by_size: dict[str, dict[str, dict[str, Any]]] = {}
    for row in manifest:
        by_size.setdefault(str(row["dataset_tag"]), {})[str(row["arch"])] = row

    rows: list[dict[str, Any]] = []
    missing: list[Path] = []

    def size_for_group(group: dict[str, dict[str, Any]]) -> float:
        first_row = next(iter(group.values()))
        return row_dataset_size(first_row, args.x_field)

    for dataset_tag, arch_rows in sorted(by_size.items(), key=lambda item: size_for_group(item[1])):
        for arch_a, arch_b in pairs:
            if arch_a not in arch_rows or arch_b not in arch_rows:
                continue

            row_a = arch_rows[arch_a]
            row_b = arch_rows[arch_b]
            path_a = sample_path(row_a, sample_root, manifest_dir, args.seed)
            path_b = sample_path(row_b, sample_root, manifest_dir, args.seed)
            if not path_a.exists() or not path_b.exists():
                missing.extend(p for p in (path_a, path_b) if not p.exists())
                if args.skip_missing:
                    print(f"SKIP missing samples for {dataset_tag} {arch_a}:{arch_b}")
                    continue
                raise FileNotFoundError(f"Missing sample file for {dataset_tag} {arch_a}:{arch_b}: {path_a}, {path_b}")

            samples = {
                arch_a: as_nchw(load_npy(path_a)),
                arch_b: as_nchw(load_npy(path_b)),
            }
            pair_metrics = reproducibility_summary(samples, nbins=args.nbins_pk)[0]
            error, score = score_from_pair_metrics(pair_metrics)
            rows.append(
                {
                    "dataset_tag": dataset_tag,
                    "dataset_size": row_dataset_size(row_a, args.x_field),
                    "pair": f"{arch_a}_vs_{arch_b}",
                    "run_a": row_a["run_name"],
                    "run_b": row_b["run_name"],
                    "sample_a": str(path_a),
                    "sample_b": str(path_b),
                    "reproducibility_error": error,
                    "reproducibility_score": score,
                    **pair_metrics,
                }
            )

    if not rows:
        msg = "No pairwise rows computed."
        if missing:
            msg += " Missing files included: " + ", ".join(str(p) for p in missing[:5])
        raise SystemExit(msg)

    write_rows_csv(rows, Path(args.output_csv))
    plot_rows(
        rows,
        pairs=pairs,
        path=Path(args.output_figure),
        memorization_max=args.memorization_max,
        generalization_min=args.generalization_min,
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_figure}")


if __name__ == "__main__":
    main()
