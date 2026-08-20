#!/usr/bin/env python
"""Plot calibration across all ten conditional-UNet training-set sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SWEEP_NAME = "nf_conditional_bias_fresh_full_sweep_200k"
ALL_DATASET_SIZES = tuple(2**power for power in range(6, 16))
PARAM_NAMES = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")
PARAM_DISPLAY_LABELS = {
    "Omega_m": r"$\Omega_\mathrm{m}$",
    "sigma_8": r"$\sigma_8$",
    "A_SN1": r"$A_\mathrm{SN1}$",
    "A_AGN1": r"$A_\mathrm{AGN1}$",
    "A_SN2": r"$A_\mathrm{SN2}$",
    "A_AGN2": r"$A_\mathrm{AGN2}$",
}


def validate_complete_sizes(frame: pd.DataFrame) -> None:
    if "dataset_size" not in frame:
        raise ValueError("calibration table has no dataset_size column")
    present = {int(value) for value in frame["dataset_size"].dropna().unique()}
    expected = set(ALL_DATASET_SIZES)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        raise ValueError(f"missing dataset sizes: {missing}")
    if extra:
        raise ValueError(f"unexpected dataset sizes: {extra}")


def _single_protocol(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "guidance_label" in out and out["guidance_label"].nunique() > 1:
        if "noguidance" not in set(out["guidance_label"]):
            raise ValueError("multiple guidance protocols found and no noguidance baseline is available")
        out = out[out["guidance_label"] == "noguidance"].copy()
    if "cfg_dropout" in out and out["cfg_dropout"].nunique() > 1:
        out = out[np.isclose(out["cfg_dropout"].astype(float), 0.0)].copy()
    return out


def plot_omega_m_grid(points: pd.DataFrame, slopes: pd.DataFrame, out: Path) -> Path:
    points = _single_protocol(points)
    slopes = _single_protocol(slopes)
    validate_complete_sizes(points)
    validate_complete_sizes(slopes)
    omega = points[points["parameter"] == "Omega_m"].copy()
    omega_slopes = slopes[slopes["parameter"] == "Omega_m"].copy()
    if omega.empty or omega_slopes.empty:
        raise ValueError("Omega_m calibration rows are missing")

    lo = float(min(omega["theta_in"].min(), omega["theta_rec_q16"].min()))
    hi = float(max(omega["theta_in"].max(), omega["theta_rec_q84"].max()))
    pad = 0.06 * max(hi - lo, 1.0e-6)
    lo, hi = lo - pad, hi + pad
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(ALL_DATASET_SIZES)))
    fig, axes = plt.subplots(2, 5, figsize=(18.0, 7.2), sharex=True, sharey=True)
    for ax, dataset_size, color in zip(axes.ravel(), ALL_DATASET_SIZES, colors):
        sub = omega[omega["dataset_size"] == dataset_size].sort_values("theta_in")
        row = omega_slopes[omega_slopes["dataset_size"] == dataset_size]
        if sub.empty or len(row) != 1:
            raise ValueError(f"expected one complete Omega_m result for N={dataset_size}")
        y = sub["theta_rec_median"].to_numpy(float)
        yerr = np.vstack(
            [
                y - sub["theta_rec_q16"].to_numpy(float),
                sub["theta_rec_q84"].to_numpy(float) - y,
            ]
        )
        slope = float(row["slope"].iloc[0])
        intercept = float(row["intercept"].iloc[0])
        ax.plot([lo, hi], [lo, hi], color="0.35", ls="--", lw=1.5, zorder=1)
        ax.errorbar(
            sub["theta_in"],
            y,
            yerr=yerr,
            fmt="o",
            ms=4.2,
            lw=1.1,
            capsize=2.0,
            color=color,
            ecolor=color,
            alpha=0.88,
            zorder=2,
        )
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color=color, lw=2.2)
        exponent = int(round(np.log2(dataset_size)))
        ax.set_title(rf"$N_{{2D}}=2^{{{exponent}}}$", fontsize=13, pad=5)
        ax.text(
            0.05,
            0.93,
            rf"slope $={slope:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=10)
    for ax in axes[-1]:
        ax.set_xlabel(r"Requested $\Omega_m$", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Recovered $\Omega_m$", fontsize=12)
    fig.suptitle(
        r"Fresh conditional UNet-128 calibration across training-set size",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        "All generators start from clean initializations and use the same 200k-update target, "
        "full six-parameter conditioning, "
        "heldout CAMELS simulations 900-931, and frozen VGG16+MLP probe.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="0.30",
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.09, top=0.88, wspace=0.12, hspace=0.22)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_omega_m_transition(points: pd.DataFrame, slopes: pd.DataFrame, out: Path) -> Path:
    """Show the full calibration transition without privileging two endpoints."""

    points = _single_protocol(points)
    slopes = _single_protocol(slopes)
    validate_complete_sizes(points)
    validate_complete_sizes(slopes)
    omega = points[points["parameter"] == "Omega_m"].copy()
    omega_slopes = slopes[slopes["parameter"] == "Omega_m"].copy()
    if omega.empty or omega_slopes.empty:
        raise ValueError("Omega_m calibration rows are missing")

    lo = float(min(omega["theta_in"].min(), omega["theta_rec_q16"].min()))
    hi = float(max(omega["theta_in"].max(), omega["theta_rec_q84"].max()))
    pad = 0.06 * max(hi - lo, 1.0e-6)
    lo, hi = lo - pad, hi + pad
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(ALL_DATASET_SIZES)))

    fig, (response_ax, slope_ax) = plt.subplots(
        1,
        2,
        figsize=(14.8, 5.5),
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )
    response_ax.plot([lo, hi], [lo, hi], color="0.25", ls="--", lw=2.0, label="ideal response")

    ordered_slopes = []
    for dataset_size, color in zip(ALL_DATASET_SIZES, colors):
        sub = omega[omega["dataset_size"] == dataset_size].sort_values("theta_in")
        row = omega_slopes[omega_slopes["dataset_size"] == dataset_size]
        if sub.empty or len(row) != 1:
            raise ValueError(f"expected one complete Omega_m result for N={dataset_size}")
        slope = float(row["slope"].iloc[0])
        intercept = float(row["intercept"].iloc[0])
        exponent = int(round(np.log2(dataset_size)))
        response_ax.scatter(
            sub["theta_in"],
            sub["theta_rec_median"],
            s=10,
            color=color,
            alpha=0.20,
            edgecolors="none",
        )
        response_ax.plot(
            [lo, hi],
            [slope * lo + intercept, slope * hi + intercept],
            color=color,
            lw=2.0,
            label=rf"$2^{{{exponent}}}$",
        )
        ordered_slopes.append(row.iloc[0])

    response_ax.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel=r"Requested $\Omega_m$",
        ylabel=r"Recovered $\Omega_m$",
        title=r"Calibration response for every $N_{2D}$",
    )
    response_ax.grid(alpha=0.18)
    response_ax.legend(
        title=r"Training images $N_{2D}$",
        ncol=2,
        fontsize=9,
        title_fontsize=9.5,
        loc="upper left",
        frameon=False,
    )

    slope_table = pd.DataFrame(ordered_slopes).sort_values("dataset_size")
    x = np.log2(slope_table["dataset_size"].to_numpy(float))
    y = slope_table["slope"].to_numpy(float)
    lower = y - slope_table["slope_ci16"].to_numpy(float)
    upper = slope_table["slope_ci84"].to_numpy(float) - y
    slope_ax.axhline(1.0, color="0.25", ls="--", lw=2.0, label="ideal slope")
    slope_ax.plot(x, y, color="0.55", lw=1.5, zorder=1)
    for x_value, y_value, low, high, color in zip(x, y, lower, upper, colors):
        slope_ax.errorbar(
            x_value,
            y_value,
            yerr=np.array([[low], [high]]),
            fmt="o",
            color=color,
            ecolor=color,
            ms=6,
            capsize=3,
            lw=1.4,
            zorder=2,
        )
    slope_ax.set_xticks(np.arange(6, 16))
    slope_ax.set_xticklabels([rf"$2^{{{power}}}$" for power in range(6, 16)], rotation=35, ha="right")
    slope_ax.set_xlabel(r"Training images $N_{2D}$")
    slope_ax.set_ylabel(r"Recovered-vs-requested $\Omega_m$ slope")
    slope_ax.set_title(r"Transition toward unit response")
    slope_ax.grid(alpha=0.18)
    slope_ax.legend(frameon=False, loc="best")

    fig.suptitle(
        r"Fresh conditional UNet-128: $\Omega_m$ calibration across training-set size",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.945,
        "All ten generators are trained from clean initializations for 200k optimizer updates; "
        "the VGG16+MLP probe and heldout cosmologies are fixed.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.84, wspace=0.27)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_parameter_slope_summary(slopes: pd.DataFrame, out: Path) -> Path:
    slopes = _single_protocol(slopes)
    validate_complete_sizes(slopes)
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.1), sharex=True)
    x_ticks = np.arange(6, 16)
    colors = plt.get_cmap("plasma")(np.linspace(0.12, 0.86, len(PARAM_NAMES)))
    for ax, parameter, color in zip(axes.ravel(), PARAM_NAMES, colors):
        sub = slopes[slopes["parameter"] == parameter].sort_values("dataset_size")
        if tuple(int(value) for value in sub["dataset_size"]) != ALL_DATASET_SIZES:
            raise ValueError(f"parameter {parameter} does not contain exactly the ten expected sizes")
        x = np.log2(sub["dataset_size"].to_numpy(float))
        y = sub["slope"].to_numpy(float)
        lower = y - sub["slope_ci16"].to_numpy(float)
        upper = sub["slope_ci84"].to_numpy(float) - y
        ax.axhline(1.0, color="0.30", ls="--", lw=1.5, label="ideal response")
        ax.errorbar(x, y, yerr=np.vstack([lower, upper]), color=color, marker="o", ms=5.0, lw=2.0, capsize=3)
        ax.set_title(PARAM_DISPLAY_LABELS.get(parameter, parameter), fontsize=15)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([rf"$2^{{{value}}}$" for value in x_ticks], rotation=35, ha="right")
        ax.grid(alpha=0.20)
        ax.tick_params(labelsize=10.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Recovered-vs-requested slope", fontsize=11.5)
    for ax in axes[-1]:
        ax.set_xlabel(r"Training images $N_{2D}$", fontsize=11.5)
    fig.suptitle("Conditional response versus training-set size", fontsize=18, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.952,
        "Points show fitted slopes; bars are 16th-84th percentile bootstrap intervals over heldout cosmologies.",
        ha="center",
        fontsize=10.5,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.11, top=0.88, wspace=0.22, hspace=0.30)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--calibration-dir", type=Path)
    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()
    calibration_dir = args.calibration_dir or project_dir / "results" / SWEEP_NAME / "calibration_vgg"
    points = pd.read_csv(calibration_dir / "bias_probe_per_cosmology_points.csv")
    slopes = pd.read_csv(calibration_dir / "bias_probe_regime_slopes.csv")
    omega_out = calibration_dir / "bias_probe_omega_m_all_dataset_sizes.png"
    transition_out = calibration_dir / "bias_probe_omega_m_transition_vs_dataset_size.png"
    slopes_out = calibration_dir / "bias_probe_all_parameter_slopes_vs_dataset_size.png"
    plot_omega_m_grid(points, slopes, omega_out)
    plot_omega_m_transition(points, slopes, transition_out)
    plot_parameter_slope_summary(slopes, slopes_out)
    print(f"Wrote {omega_out}")
    print(f"Wrote {transition_out}")
    print(f"Wrote {slopes_out}")


if __name__ == "__main__":
    main()
