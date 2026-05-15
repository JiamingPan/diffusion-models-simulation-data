#!/usr/bin/env python
"""Audit and score higher post-hoc EMA samples for nf_sweep_v2.

This is meant to be run on Great Lakes after the sampling array finishes.  It
checks whether the widened EMA targets exist, computes the same one-point and
P(k) metrics used in the quick-check notebook, and prints the best target per
variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.io import as_nchw, load_real_from_config
from simdiff_eval.metrics import field_histogram, power_spectrum_summary


DEFAULT_EMA_LABELS = [
    "raw",
    "ema0p02",
    "ema0p04",
    "ema0p06",
    "ema0p08",
    "ema0p10",
    "ema0p13",
    "ema0p16",
    "ema0p20",
    "ema0p25",
]

EMA_VALUES = {
    "raw": np.nan,
    "ema0p02": 0.02,
    "ema0p04": 0.04,
    "ema0p06": 0.06,
    "ema0p08": 0.08,
    "ema0p10": 0.10,
    "ema0p13": 0.13,
    "ema0p16": 0.16,
    "ema0p20": 0.20,
    "ema0p25": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--arch", choices=["u64", "u128"], default="u128")
    parser.add_argument("--sampler", default="train_full", choices=["train_full", "dpm25", "heun50"])
    parser.add_argument("--variant", default=None, help="Optional variant_tag filter, e.g. nick_default.")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_EMA_LABELS),
        help="Comma-separated EMA labels to check.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-generated", type=int, default=32)
    parser.add_argument("--max-real-cubes", type=int, default=32)
    parser.add_argument("--max-real-hist", type=int, default=512)
    parser.add_argument("--max-real-pk", type=int, default=512)
    parser.add_argument("--pk-nbins", type=int, default=25)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def evenly_limit(arr: np.ndarray, limit: int | None) -> np.ndarray:
    arr = np.asarray(arr)
    if limit is None or len(arr) <= limit:
        return np.array(arr, copy=True)
    idx = np.linspace(0, len(arr) - 1, limit, dtype=int)
    return np.array(arr[idx], copy=True)


def load_npz_samples(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        key = "samples" if "samples" in data.files else data.files[0]
        return as_nchw(np.asarray(data[key])).copy()


def npz_n(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=True) as data:
            key = "samples" if "samples" in data.files else data.files[0]
            return int(data[key].shape[0])
    except Exception:
        return 0


def sample_path(sample_root: Path, run_name: str, seed: int, ema_label: str, sampler: str) -> Path:
    return sample_root / f"{run_name}_seed{seed}_{ema_label}_{sampler}.npz"


def data_signature(config_path: Path) -> str:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    data = cfg.get("data", {})
    keys = [
        "img_path",
        "reshape",
        "two_dim",
        "zthin",
        "n_samples",
        "seed",
        "log",
        "transform",
        "normalization",
        "norm_kwargs",
    ]
    slim = {k: data.get(k) for k in keys if k in data}
    return json.dumps(slim, sort_keys=True, default=str)


def onepoint_summary(real: np.ndarray, generated: np.ndarray, bins: int = 120) -> dict[str, float]:
    rh = field_histogram(real, bins=bins)
    gh = field_histogram(generated, bins=bins)
    edges = np.asarray(rh["bin_edges"])
    width = float(np.mean(np.diff(edges)))
    hist_l1 = float(np.sum(np.abs(np.asarray(rh["hist"]) - np.asarray(gh["hist"]))) * width)
    return {
        "generated_mean": gh["mean"],
        "generated_std": gh["std"],
        "std_ratio": gh["std"] / max(rh["std"], 1e-30),
        "hist_l1": hist_l1,
    }


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    manifest_path = project_dir / "local/nf_sweep_v2/manifest.json"
    sample_root = project_dir / "results/nf_sweep_v2/samples"
    output_dir = project_dir / "results/nf_sweep_v2/quickcheck"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    manifest = json.loads(manifest_path.read_text())
    runs = [row for row in manifest if row["arch"] == args.arch]
    if args.variant:
        runs = [row for row in runs if row["variant_tag"] == args.variant]
    if not runs:
        raise SystemExit(f"No runs matched arch={args.arch!r}, variant={args.variant!r}.")

    audit_rows: list[dict[str, Any]] = []
    for row in runs:
        for label in labels:
            path = sample_path(sample_root, row["run_name"], args.seed, label, args.sampler)
            n = npz_n(path)
            audit_rows.append(
                {
                    "run_name": row["run_name"],
                    "arch": row["arch"],
                    "variant": row["variant_tag"],
                    "sampler": args.sampler,
                    "ema_label": label,
                    "ema_value": EMA_VALUES.get(label, np.nan),
                    "n_available": n,
                    "status": "ok" if n >= args.max_generated else ("short" if n > 0 else "missing"),
                    "sample_path": str(path),
                    "config_path": str(project_dir / row["config"]),
                }
            )

    audit_df = pd.DataFrame(audit_rows)
    print("audit:", audit_df["status"].value_counts().to_dict())
    print(audit_df[["variant", "sampler", "ema_label", "n_available", "status"]].to_string(index=False))

    audit_out = output_dir / f"nf_sweep_v2_{args.arch}_{args.sampler}_higher_ema_audit.csv"
    audit_df.to_csv(audit_out, index=False)
    print("wrote", audit_out)

    if args.audit_only:
        return

    present = audit_df[audit_df["n_available"] > 0].copy()
    if present.empty:
        print("No available samples to score.")
        return

    real_cache: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    for rec in present.to_dict("records"):
        n_used = min(int(rec["n_available"]), args.max_generated)
        config_path = Path(rec["config_path"])
        sig = data_signature(config_path)
        if sig not in real_cache:
            real_cache[sig] = load_real_from_config(config_path, max_raw_samples=args.max_real_cubes)
        real = real_cache[sig]
        generated = evenly_limit(load_npz_samples(Path(rec["sample_path"])), n_used)
        onepoint = onepoint_summary(evenly_limit(real, args.max_real_hist), generated)
        pk_summary = power_spectrum_summary(
            evenly_limit(real, args.max_real_pk),
            generated,
            nbins=args.pk_nbins,
        )
        metric_rows.append(
            {
                **{k: rec[k] for k in ["run_name", "arch", "variant", "sampler", "ema_label", "ema_value"]},
                "n_used": n_used,
                **onepoint,
                **pk_summary,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["variant", "ema_value"], na_position="first")
    metrics_out = output_dir / f"nf_sweep_v2_{args.arch}_{args.sampler}_higher_ema_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print("wrote", metrics_out)

    print("\nBest by P(k) log10 MAE:")
    best_pk = metrics_df.loc[metrics_df.groupby("variant")["pk_log10_mae"].idxmin()]
    print(
        best_pk[
            ["variant", "ema_label", "n_used", "pk_log10_mae", "hist_l1", "std_ratio", "pk_ratio_low_k", "pk_ratio_mid_k", "pk_ratio_high_k"]
        ]
        .sort_values("pk_log10_mae")
        .to_string(index=False)
    )

    print("\nBest by one-point histogram L1:")
    best_hist = metrics_df.loc[metrics_df.groupby("variant")["hist_l1"].idxmin()]
    print(
        best_hist[
            ["variant", "ema_label", "n_used", "hist_l1", "pk_log10_mae", "std_ratio", "pk_ratio_low_k", "pk_ratio_mid_k", "pk_ratio_high_k"]
        ]
        .sort_values("hist_l1")
        .to_string(index=False)
    )

    high = metrics_df[metrics_df["ema_label"].isin(["ema0p13", "ema0p16", "ema0p20", "ema0p25"])]
    if len(high):
        print("\nHigher EMA targets only:")
        print(
            high[
                ["variant", "ema_label", "n_used", "pk_log10_mae", "hist_l1", "std_ratio", "pk_ratio_low_k", "pk_ratio_mid_k", "pk_ratio_high_k"]
            ]
            .sort_values(["variant", "ema_value"])
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
