#!/usr/bin/env python
"""Score small post-hoc EMA samples for one Fig. 2 generalization run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.io import _load_real_tanh_from_config, as_nchw
from simdiff_eval.metrics import batch_power_spectra, field_histogram


DEFAULT_LABELS = [
    "raw",
    "ema0p001",
    "ema0p002",
    "ema0p003",
    "ema0p004",
    "ema0p005",
    "ema0p006",
    "ema0p008",
    "ema0p01",
    "ema0p015",
    "ema0p02",
    "ema0p03",
    "ema0p04",
    "ema0p05",
]

EMA_VALUES = {
    "raw": np.nan,
    "ema0p001": 0.001,
    "ema0p002": 0.002,
    "ema0p003": 0.003,
    "ema0p004": 0.004,
    "ema0p005": 0.005,
    "ema0p006": 0.006,
    "ema0p008": 0.008,
    "ema0p01": 0.010,
    "ema0p015": 0.015,
    "ema0p02": 0.020,
    "ema0p03": 0.030,
    "ema0p04": 0.040,
    "ema0p05": 0.050,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--run-name", default="nf_fig2_u64_d2p15_noaug_200k")
    parser.add_argument("--sampler", default="train_full")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-generated", type=int, default=128)
    parser.add_argument("--max-real-cubes", type=int, default=8)
    parser.add_argument("--max-real-hist", type=int, default=512)
    parser.add_argument("--max-real-pk", type=int, default=256)
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


def load_manifest_row(manifest_path: Path, run_name: str) -> dict[str, Any]:
    rows = json.loads(manifest_path.read_text())
    matches = [row for row in rows if row["run_name"] == run_name]
    if not matches:
        raise SystemExit(f"No manifest row for run_name={run_name!r} in {manifest_path}")
    return matches[0]


def load_real_lightweight(config_path: Path, max_raw_samples: int) -> np.ndarray:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg)
    data_cfg = dict(cfg.get("data", {}))
    current = data_cfg.get("n_samples")
    data_cfg["n_samples"] = int(max_raw_samples) if current is None else min(int(current), int(max_raw_samples))
    cfg["data"] = data_cfg
    return _load_real_tanh_from_config(cfg, utils_module=None)


def onepoint_summary_from_reference(real_hist: dict[str, Any], generated: np.ndarray, bins: int = 120) -> dict[str, float]:
    gen_hist = field_histogram(generated, bins=bins)
    edges = np.asarray(real_hist["bin_edges"])
    width = float(np.mean(np.diff(edges)))
    hist_l1 = float(np.sum(np.abs(np.asarray(real_hist["hist"]) - np.asarray(gen_hist["hist"]))) * width)
    return {
        "generated_mean": gen_hist["mean"],
        "generated_std": gen_hist["std"],
        "std_ratio": gen_hist["std"] / max(real_hist["std"], 1e-30),
        "hist_l1": hist_l1,
    }


def pk_summary_from_reference(real_mean: np.ndarray, generated: np.ndarray, nbins: int) -> dict[str, float]:
    pk_gen, _ = batch_power_spectra(generated, nbins=nbins)
    gen_mean = np.nanmean(pk_gen, axis=0)
    ratio = gen_mean / np.clip(real_mean, 1e-30, None)
    log_abs = np.abs(np.log10(np.clip(ratio, 1e-30, None)))

    finite = np.where(np.isfinite(ratio))[0]
    thirds = np.array_split(finite, 3) if len(finite) else [[], [], []]
    band_means = [
        float(np.nanmean(ratio[idx])) if len(idx) else float("nan")
        for idx in thirds
    ]
    return {
        "pk_log10_mae": float(np.nanmean(log_abs)),
        "pk_ratio_low_k": band_means[0],
        "pk_ratio_mid_k": band_means[1],
        "pk_ratio_high_k": band_means[2],
    }


def plot_metrics(metrics_df: pd.DataFrame, output_path: Path, title: str) -> None:
    raw = metrics_df[metrics_df["ema_label"] == "raw"]
    ema = metrics_df[metrics_df["ema_label"] != "raw"].sort_values("ema_value")
    if ema.empty:
        return

    raw_x = float(ema["ema_value"].min()) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for ax, metric, ylabel, panel_title in [
        (axes[0], "hist_l1", "one-point histogram L1", "one-point error, lower is better"),
        (axes[1], "pk_log10_mae", "P(k) log10 MAE", "P(k) error, lower is better"),
    ]:
        ax.plot(ema["ema_value"], ema[metric], marker="o", color="black", label="post-hoc EMA")
        if len(raw):
            ax.scatter([raw_x], [float(raw.iloc[0][metric])], marker="*", s=180, color="black", label="raw")
        ax.set_xscale("log")
        ax.set_xlabel("EMA sigma_rel (log scale; star = raw checkpoint)")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    manifest_path = args.manifest or project_dir / "local/nf_generalize_fig2/manifest.json"
    sample_root = project_dir / "results/nf_generalize_fig2/samples"
    output_dir = project_dir / "results/nf_generalize_fig2/quickcheck"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    row = load_manifest_row(manifest_path, args.run_name)
    config_path = project_dir / row["config"]

    audit_rows: list[dict[str, Any]] = []
    for label in labels:
        path = sample_path(sample_root, args.run_name, args.seed, label, args.sampler)
        n_available = npz_n(path)
        audit_rows.append(
            {
                "run_name": args.run_name,
                "arch": row.get("arch"),
                "dataset_tag": row.get("dataset_tag"),
                "dataset_size": row.get("dataset_size"),
                "sampler": args.sampler,
                "ema_label": label,
                "ema_value": EMA_VALUES.get(label, np.nan),
                "n_available": n_available,
                "status": "ok" if n_available >= args.max_generated else ("short" if n_available > 0 else "missing"),
                "sample_path": str(path),
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    audit_out = output_dir / f"{args.run_name}_{args.sampler}_small_ema_audit.csv"
    audit_df.to_csv(audit_out, index=False)
    print("audit:", audit_df["status"].value_counts().to_dict(), flush=True)
    print(audit_df[["ema_label", "n_available", "status", "sample_path"]].to_string(index=False), flush=True)
    print("wrote", audit_out, flush=True)
    if args.audit_only:
        return

    present = audit_df[audit_df["n_available"] > 0].copy()
    if present.empty:
        print("No available samples to score.", flush=True)
        return

    print(
        f"building real reference from {config_path.name}: "
        f"max_real_cubes={args.max_real_cubes}, max_real_hist={args.max_real_hist}, max_real_pk={args.max_real_pk}",
        flush=True,
    )
    real = load_real_lightweight(config_path, max_raw_samples=args.max_real_cubes)
    real_hist = field_histogram(evenly_limit(real, args.max_real_hist), bins=120)
    pk_real, _ = batch_power_spectra(evenly_limit(real, args.max_real_pk), nbins=args.pk_nbins)
    real_pk_mean = np.nanmean(pk_real, axis=0)

    metric_rows: list[dict[str, Any]] = []
    for rec in present.to_dict("records"):
        n_used = min(int(rec["n_available"]), args.max_generated)
        print(f"scoring {rec['ema_label']} n_used={n_used}", flush=True)
        generated = evenly_limit(load_npz_samples(Path(rec["sample_path"])), n_used)
        metric_rows.append(
            {
                **rec,
                "n_used": n_used,
                **onepoint_summary_from_reference(real_hist, generated),
                **pk_summary_from_reference(real_pk_mean, generated, nbins=args.pk_nbins),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("ema_value", na_position="first")
    metrics_out = output_dir / f"{args.run_name}_{args.sampler}_small_ema_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print("wrote", metrics_out, flush=True)

    best_pk = metrics_df.sort_values(["pk_log10_mae", "hist_l1"]).iloc[0]
    best_hist = metrics_df.sort_values(["hist_l1", "pk_log10_mae"]).iloc[0]
    print("\nBest by P(k) log10 MAE:", flush=True)
    print(best_pk[["ema_label", "n_used", "pk_log10_mae", "hist_l1", "std_ratio", "pk_ratio_low_k", "pk_ratio_mid_k", "pk_ratio_high_k"]].to_string(), flush=True)
    print("\nBest by one-point histogram L1:", flush=True)
    print(best_hist[["ema_label", "n_used", "hist_l1", "pk_log10_mae", "std_ratio", "pk_ratio_low_k", "pk_ratio_mid_k", "pk_ratio_high_k"]].to_string(), flush=True)

    plot_path = output_dir / f"{args.run_name}_{args.sampler}_small_ema_sweep.png"
    plot_metrics(
        metrics_df,
        plot_path,
        f"{row.get('arch', '')}: small EMA sweep for {args.run_name} ({args.sampler})",
    )
    print("wrote", plot_path, flush=True)


if __name__ == "__main__":
    main()
