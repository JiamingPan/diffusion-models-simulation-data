#!/usr/bin/env python
"""Plot VGG encoder R^2 comparisons and VGG calibration slopes.

This is a lightweight presentation helper for the continuous HI cosmology bias
probe.  It reads the VGG encoder job logs and the generated-sample calibration
CSV, then writes compact comparison figures with the R^2/slope values printed on
the plot.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARAM_ORDER = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
PARAM_LABELS = {
    "Omega_m": r"$\Omega_m$",
    "sigma_8": r"$\sigma_8$",
    "A_SN1": r"$A_{\rm SN1}$",
    "A_AGN1": r"$A_{\rm AGN1}$",
    "A_SN2": r"$A_{\rm SN2}$",
    "A_AGN2": r"$A_{\rm AGN2}$",
}

DEFAULT_ENCODER_LOGS = {
    "smoke: avg+max + MLP(64), 512 train slices [51475729]": "logs/nf_conditional_bias_probe/vgg_encoder_51475729.out",
    "default: avg+max + MLP(512,256), 32k train slices [51475731]": "logs/nf_conditional_bias_probe/vgg_encoder_51475731.out",
    "best: avg+max + MLP(1024,512,256), 65k train slices [51475738]": "logs/nf_conditional_bias_probe/vgg_encoder_51475738.out",
    "ablation: avg + MLP(1024,512,256), 65k train slices [51475739]": "logs/nf_conditional_bias_probe/vgg_encoder_51475739.out",
    "linear check: avg+max + Ridge(alpha=1), 65k train slices [51475740]": "logs/nf_conditional_bias_probe/vgg_encoder_51475740.out",
}


def parse_metric_lines(path: Path, label: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"^\s*test\s+per_cosmology\s+"
        r"(?P<parameter>Omega_m|sigma_8|A_SN1|A_AGN1|A_SN2|A_AGN2)\s+"
        r".*?\s(?P<n>\d+)\s+"
        r"(?P<mae>[-+]?\d*\.?\d+)\s+"
        r"(?P<rmse>[-+]?\d*\.?\d+)\s+"
        r"(?P<bias>[-+]?\d*\.?\d+)\s+"
        r"(?P<r2>[-+]?\d*\.?\d+)\s*$"
    )
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        row = match.groupdict()
        rows.append(
            {
                "encoder": label,
                "parameter": row["parameter"],
                "n": int(row["n"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "bias": float(row["bias"]),
                "r2": float(row["r2"]),
                "log_path": str(path),
            }
        )
    return rows


def collect_encoder_metrics(project_dir: Path, log_specs: list[str]) -> pd.DataFrame:
    specs: dict[str, str] = {}
    if log_specs:
        for spec in log_specs:
            if "=" not in spec:
                raise ValueError(f"Expected LABEL=PATH for --encoder-log, got {spec!r}")
            label, path = spec.split("=", 1)
            specs[label] = path
    else:
        specs = DEFAULT_ENCODER_LOGS

    rows: list[dict[str, object]] = []
    for label, rel_path in specs.items():
        rows.extend(parse_metric_lines(project_dir / rel_path, label))
    if not rows:
        raise FileNotFoundError("No VGG encoder metrics found in requested logs.")
    df = pd.DataFrame(rows)
    df["parameter"] = pd.Categorical(df["parameter"], PARAM_ORDER, ordered=True)
    return df.sort_values(["parameter", "encoder"]).reset_index(drop=True)


def plot_r2(df: pd.DataFrame, out: Path) -> None:
    encoders = list(dict.fromkeys(df["encoder"].astype(str)))
    params = PARAM_ORDER
    x = np.arange(len(params), dtype=float)
    width = min(0.18, 0.72 / max(len(encoders), 1))
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(14.8, 6.2), constrained_layout=False)
    for i, encoder in enumerate(encoders):
        sub = df[df["encoder"] == encoder].set_index("parameter")
        vals = np.array([float(sub.loc[p, "r2"]) if p in sub.index else np.nan for p in params])
        xpos = x + (i - (len(encoders) - 1) / 2) * width
        bars = ax.bar(xpos, vals, width=width, label=encoder, color=colors[i % len(colors)], alpha=0.92)
        for bar, val in zip(bars, vals):
            if not np.isfinite(val):
                continue
            y = val + (0.025 if val >= 0 else -0.045)
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=9,
                rotation=90,
            )

    ax.axhline(0, color="0.2", lw=1.1)
    ax.axhline(1, color="0.45", lw=1.0, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([PARAM_LABELS[p] for p in params], fontsize=13)
    ax.set_ylabel(r"Real held-out $R^2$", fontsize=14)
    ax.set_title(
        "Frozen VGG16 encoder comparison on real held-out HI fields",
        fontsize=18,
        pad=14,
    )
    ax.set_ylim(-0.12, 1.02)
    ax.grid(axis="y", alpha=0.22)
    ax.text(
        0.01,
        0.98,
        "Each bar uses frozen VGG16 features; only the regression head is trained.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="0.25",
    )
    ax.legend(
        title="Encoder variant",
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.07, right=0.60, bottom=0.13, top=0.88)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_slopes(slopes: pd.DataFrame, out: Path) -> None:
    focus = slopes[slopes["parameter"].isin(["Omega_m", "sigma_8"])].copy()
    if focus.empty:
        return
    focus["parameter"] = pd.Categorical(focus["parameter"], ["Omega_m", "sigma_8"], ordered=True)
    focus = focus.sort_values(["parameter", "dataset_size"])

    regimes = list(dict.fromkeys(focus["regime"].astype(str)))
    params = ["Omega_m", "sigma_8"]
    x = np.arange(len(params), dtype=float)
    width = 0.28
    colors = {"memorization": "#D62728", "generalization": "#1F77B4"}

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for i, regime in enumerate(regimes):
        sub = focus[focus["regime"] == regime].set_index("parameter")
        vals = np.array([float(sub.loc[p, "slope"]) if p in sub.index else np.nan for p in params])
        lo = np.array([float(sub.loc[p, "slope_ci16"]) if p in sub.index else np.nan for p in params])
        hi = np.array([float(sub.loc[p, "slope_ci84"]) if p in sub.index else np.nan for p in params])
        yerr = np.vstack([vals - lo, hi - vals])
        xpos = x + (i - (len(regimes) - 1) / 2) * width
        ax.bar(xpos, vals, width=width, color=colors.get(regime, "0.5"), alpha=0.92, label=regime)
        ax.errorbar(xpos, vals, yerr=yerr, fmt="none", ecolor="0.2", capsize=3, lw=1.4)
        for xx, val in zip(xpos, vals):
            if np.isfinite(val):
                ax.text(xx, val + 0.035, f"{val:.2f}", ha="center", va="bottom", fontsize=11)

    ax.axhline(1, color="0.25", ls="--", lw=1.4, label="ideal slope = 1")
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([PARAM_LABELS[p] for p in params], fontsize=14)
    ax.set_ylabel("Recovered vs input slope", fontsize=14)
    ax.set_title("VGG encoder calibration: generated fields track input cosmology", fontsize=17)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument(
        "--encoder-log",
        action="append",
        default=[],
        help="Optional LABEL=PATH relative to project dir. May be repeated.",
    )
    parser.add_argument(
        "--slopes",
        default="results/nf_conditional_bias_probe/calibration_vgg/bias_probe_regime_slopes.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/nf_conditional_bias_probe/encoder",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = project_dir / args.output_dir

    metrics = collect_encoder_metrics(project_dir, args.encoder_log)
    metrics_path = output_dir / "vgg_encoder_r2_comparison.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    r2_plot = output_dir / "vgg_encoder_r2_comparison.png"
    plot_r2(metrics, r2_plot)

    print("VGG encoder R2 comparison:")
    print(metrics.pivot(index="parameter", columns="encoder", values="r2").round(4).to_string())
    print(f"wrote {metrics_path}")
    print(f"wrote {r2_plot}")

    slopes_path = project_dir / args.slopes
    if slopes_path.exists():
        slopes = pd.read_csv(slopes_path)
        slope_plot = project_dir / "results" / "nf_conditional_bias_probe" / "calibration_vgg" / "bias_probe_vgg_main_slopes.png"
        plot_slopes(slopes, slope_plot)
        print("VGG generated-sample slopes:")
        print(slopes[slopes["parameter"].isin(["Omega_m", "sigma_8"])].round(4).to_string(index=False))
        print(f"wrote {slope_plot}")
    else:
        print(f"missing slopes CSV: {slopes_path}")


if __name__ == "__main__":
    main()
