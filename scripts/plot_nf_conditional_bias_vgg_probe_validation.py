#!/usr/bin/env python
"""Plot held-out-real VGG probe slopes and R2 for all CAMELS parameters."""

from __future__ import annotations

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
PARAM_COLORS = ["#4C2A85", "#375A9E", "#287C8E", "#28A889", "#76C85A", "#C6D800"]


def _required_columns(parameter: str) -> tuple[str, ...]:
    return (
        f"{parameter}_true",
        f"{parameter}_pred_median",
        f"{parameter}_pred_q16",
        f"{parameter}_pred_q84",
    )


def _finite_parameter_arrays(
    predictions: pd.DataFrame,
    parameter: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [column for column in _required_columns(parameter) if column not in predictions.columns]
    if missing:
        raise ValueError(f"Missing held-out probe columns for {parameter}: {missing}")

    x = predictions[f"{parameter}_true"].to_numpy(float)
    y = predictions[f"{parameter}_pred_median"].to_numpy(float)
    q16 = predictions[f"{parameter}_pred_q16"].to_numpy(float)
    q84 = predictions[f"{parameter}_pred_q84"].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(q16) & np.isfinite(q84)
    if finite.sum() < 3:
        raise ValueError(f"Need at least three finite held-out cosmologies for {parameter}")
    return x[finite], y[finite], q16[finite], q84[finite]


def compute_probe_validation_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fit response slopes and compute R2 from the same per-cosmology medians."""
    rows: list[dict[str, float | int | str]] = []
    for parameter in PARAM_ORDER:
        x, y, _, _ = _finite_parameter_arrays(predictions, parameter)
        slope, intercept = np.polyfit(x, y, 1)
        denominator = float(np.sum((x - x.mean()) ** 2))
        if denominator <= 0:
            raise ValueError(f"Held-out truth values have zero variance for {parameter}")
        r2 = 1.0 - float(np.sum((y - x) ** 2)) / denominator
        rows.append(
            {
                "parameter": parameter,
                "n_heldout": int(len(x)),
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
            }
        )
    return pd.DataFrame(rows)


def _limits(x: np.ndarray, q16: np.ndarray, q84: np.ndarray) -> tuple[float, float]:
    lo = float(min(x.min(), q16.min()))
    hi = float(max(x.max(), q84.max()))
    pad = 0.08 * max(hi - lo, 1.0e-6)
    return lo - pad, hi + pad


def plot_probe_one_to_one(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
    *,
    probe_label: str,
) -> Path:
    lookup = summary.set_index("parameter")
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.4), constrained_layout=False)
    for index, (axis, parameter) in enumerate(zip(axes.ravel(), PARAM_ORDER)):
        x, y, q16, q84 = _finite_parameter_arrays(predictions, parameter)
        row = lookup.loc[parameter]
        lo, hi = _limits(x, q16, q84)
        axis.plot([lo, hi], [lo, hi], "--", color="0.30", lw=1.8, label="ideal response")
        xs = np.array([lo, hi])
        axis.plot(
            xs,
            float(row["slope"]) * xs + float(row["intercept"]),
            color=PARAM_COLORS[index],
            lw=2.4,
            label="fitted response",
        )
        yerr = np.vstack([np.maximum(y - q16, 0.0), np.maximum(q84 - y, 0.0)])
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            ms=5.5,
            capsize=2.5,
            color=PARAM_COLORS[index],
            ecolor=PARAM_COLORS[index],
            alpha=0.88,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_title(PARAM_LABELS[parameter], fontsize=17, pad=8)
        axis.set_xlabel("True parameter", fontsize=12)
        axis.set_ylabel("Probe recovery", fontsize=12)
        axis.text(
            0.04,
            0.95,
            rf"slope = {float(row['slope']):.2f}" + "\n" + rf"$R^2$ = {float(row['r2']):.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color=PARAM_COLORS[index],
            fontweight="semibold",
        )
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=10.5)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    fig.suptitle("Frozen VGG16 probe on real held-out CAMELS maps", fontsize=20, y=0.985)
    fig.text(0.5, 0.947, probe_label, ha="center", va="center", fontsize=10.5, color="0.35")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.86, wspace=0.27, hspace=0.34)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_probe_summary(summary: pd.DataFrame, out: Path, *, probe_label: str) -> Path:
    ordered = summary.set_index("parameter").loc[PARAM_ORDER].reset_index()
    x = np.arange(len(PARAM_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=False)
    for axis, metric, title in zip(
        axes,
        ("slope", "r2"),
        ("Held-out response slope", r"Held-out $R^2$"),
    ):
        values = ordered[metric].to_numpy(float)
        axis.axhline(1.0, color="0.3", ls="--", lw=1.6, label="ideal = 1")
        axis.axhline(0.0, color="0.55", lw=0.9)
        axis.plot(x, values, color="0.58", lw=1.4, zorder=1)
        axis.scatter(x, values, c=PARAM_COLORS, s=76, zorder=2)
        for xpos, value in zip(x, values):
            axis.annotate(f"{value:.2f}", (xpos, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
        axis.set_xticks(x, [PARAM_LABELS[p] for p in PARAM_ORDER])
        axis.set_title(title, fontsize=16, pad=9)
        axis.grid(axis="y", alpha=0.18)
        axis.tick_params(labelsize=11)
        finite = values[np.isfinite(values)]
        lower = min(-0.1, float(finite.min()) - 0.12)
        upper = max(1.05, float(finite.max()) + 0.12)
        axis.set_ylim(lower, upper)
    axes[0].set_ylabel("Metric value", fontsize=12)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("What the frozen probe can recover before testing generated maps", fontsize=19, y=0.98)
    fig.text(0.5, 0.925, probe_label, ha="center", fontsize=10.5, color="0.35")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.82, wspace=0.18)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def write_probe_validation_outputs(
    predictions: pd.DataFrame,
    *,
    panel_path: Path,
    summary_path: Path,
    table_path: Path,
    probe_label: str,
) -> pd.DataFrame:
    summary = compute_probe_validation_summary(predictions)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    plot_probe_one_to_one(predictions, summary, panel_path, probe_label=probe_label)
    plot_probe_summary(summary, summary_path, probe_label=probe_label)
    return summary

