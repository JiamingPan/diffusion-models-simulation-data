#!/usr/bin/env python
"""Build paper-ready conditional-recovery and companion verification figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "ai4science_verification"
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

from paperstyle import FULL_W, INK, set_paper_style, style_axis


ALL_POWERS = tuple(range(6, 16))
ALL_DATASET_SIZES = tuple(2**power for power in ALL_POWERS)
REPRESENTATIVE_POWERS = (6, 11, 15)
REPRESENTATIVE_MARKERS = ("o", "s", "^")
PARAM_ORDER = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")
PARAM_LABELS = {
    "Omega_m": r"$\Omega_m$",
    "sigma_8": r"$\sigma_8$",
    "A_SN1": r"$A_{\rm SN1}$",
    "A_AGN1": r"$A_{\rm AGN1}$",
    "A_SN2": r"$A_{\rm SN2}$",
    "A_AGN2": r"$A_{\rm AGN2}$",
}
ARCHITECTURES = (
    ("u64", "U-Net-64", "#009E73", "^"),
    ("u128", "U-Net-128", "#D55E00", "o"),
    ("u256", "U-Net-256", "#0072B2", "s"),
)


def _single_protocol(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "guidance_label" in out and out["guidance_label"].nunique() > 1:
        if "noguidance" not in set(out["guidance_label"]):
            raise ValueError("multiple guidance protocols found without a noguidance baseline")
        out = out[out["guidance_label"] == "noguidance"].copy()
    if "cfg_dropout" in out and out["cfg_dropout"].nunique() > 1:
        out = out[np.isclose(out["cfg_dropout"].astype(float), 0.0)].copy()
    return out


def _require_sizes(frame: pd.DataFrame) -> None:
    if "dataset_size" not in frame:
        raise ValueError("table has no dataset_size column")
    present = {int(value) for value in frame["dataset_size"].dropna().unique()}
    missing = sorted(set(ALL_DATASET_SIZES) - present)
    extra = sorted(present - set(ALL_DATASET_SIZES))
    if missing:
        raise ValueError(f"missing dataset sizes: {missing}")
    if extra:
        raise ValueError(f"unexpected dataset sizes: {extra}")


def _training_colors() -> dict[int, np.ndarray]:
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(ALL_POWERS)))
    return dict(zip(ALL_DATASET_SIZES, colors))


def style_training_size_axis(axis) -> None:
    """Use the exact shared horizontal power-of-two axis for paper comparisons."""

    axis.set_xlim(5.65, 15.35)
    axis.set_xticks(ALL_POWERS)
    axis.set_xticklabels([rf"$2^{{{power}}}$" for power in ALL_POWERS], rotation=0)
    style_axis(axis)


def _omega_tables(
    points: pd.DataFrame,
    slopes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = _single_protocol(points)
    slopes = _single_protocol(slopes)
    omega_points = points[points["parameter"] == "Omega_m"].copy()
    omega_slopes = slopes[slopes["parameter"] == "Omega_m"].copy()
    _require_sizes(omega_points)
    _require_sizes(omega_slopes)
    counts = omega_slopes.groupby("dataset_size", observed=True).size()
    invalid = counts[counts != 1]
    if not invalid.empty:
        raise ValueError(f"expected one Omega_m fit per size, found {invalid.to_dict()}")
    return omega_points, omega_slopes.sort_values("dataset_size").reset_index(drop=True)


def build_conditional_recovery_figure(
    points: pd.DataFrame,
    slopes: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Return the three-regime calibration view and full transition curve."""

    set_paper_style()
    omega_points, report = _omega_tables(points, slopes)
    required = {
        "theta_in",
        "theta_rec_median",
        "theta_rec_q16",
        "theta_rec_q84",
    }
    missing = sorted(required - set(omega_points.columns))
    if missing:
        raise ValueError(f"Omega_m point table is missing columns: {missing}")

    lo = float(min(omega_points["theta_in"].min(), omega_points["theta_rec_q16"].min()))
    hi = float(max(omega_points["theta_in"].max(), omega_points["theta_rec_q84"].max()))
    pad = 0.055 * max(hi - lo, 1.0e-6)
    limits = (lo - pad, hi + pad)
    colors = _training_colors()

    figure = plt.figure(figsize=(FULL_W, 4.40))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(1.35, 1.0),
        left=0.090,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.28,
        hspace=0.58,
    )
    scatter_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    transition_axis = figure.add_subplot(grid[1, :])

    for index, (axis, power, marker) in enumerate(
        zip(scatter_axes, REPRESENTATIVE_POWERS, REPRESENTATIVE_MARKERS)
    ):
        dataset_size = 2**power
        sub = omega_points[omega_points["dataset_size"] == dataset_size].sort_values("theta_in")
        row = report[report["dataset_size"] == dataset_size].iloc[0]
        x = sub["theta_in"].to_numpy(float)
        y = sub["theta_rec_median"].to_numpy(float)
        q16 = sub["theta_rec_q16"].to_numpy(float)
        q84 = sub["theta_rec_q84"].to_numpy(float)
        yerr = np.vstack((np.maximum(y - q16, 0.0), np.maximum(q84 - y, 0.0)))
        color = colors[dataset_size]
        axis.plot(limits, limits, color="0.38", ls="--", lw=0.9, label="_nolegend_")
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            ms=2.8,
            capsize=1.4,
            elinewidth=0.7,
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=0.9,
        )
        fit_x = np.asarray(limits)
        axis.plot(
            fit_x,
            float(row["slope"]) * fit_x + float(row["intercept"]),
            color=color,
            lw=1.35,
        )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.text(
            0.05,
            0.95,
            rf"$N_{{2D}}=2^{{{power}}}$"
            + "\n"
            + rf"slope = {float(row['slope']):.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            color=INK,
        )
        if index == 0:
            axis.set_ylabel(r"Recovered $\Omega_m$")
        else:
            axis.tick_params(labelleft=False)
        axis.set_xlabel("")
        style_axis(axis)

    x = np.log2(report["dataset_size"].to_numpy(float))
    y = report["slope"].to_numpy(float)
    lower = np.maximum(y - report["slope_ci16"].to_numpy(float), 0.0)
    upper = np.maximum(report["slope_ci84"].to_numpy(float) - y, 0.0)
    transition_axis.axhline(1.0, color="0.38", ls="--", lw=0.9, label="_nolegend_")
    transition_axis.plot(x, y, color="0.55", lw=0.8, zorder=1)
    for xpos, value, low, high, dataset_size in zip(
        x, y, lower, upper, report["dataset_size"].astype(int)
    ):
        transition_axis.errorbar(
            xpos,
            value,
            yerr=np.array([[low], [high]]),
            fmt="o",
            ms=3.7,
            capsize=1.8,
            elinewidth=0.8,
            color=colors[dataset_size],
            ecolor=colors[dataset_size],
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=2,
        )
    style_training_size_axis(transition_axis)
    transition_axis.set_xlabel(r"Training images $N_{2D}$")
    transition_axis.set_ylabel(r"$\Omega_m$ response slope", labelpad=2)
    transition_axis.set_ylim(bottom=min(0.0, float(report["slope_ci16"].min()) - 0.04), top=1.05)
    scatter_left = scatter_axes[0].get_position().x0
    scatter_right = scatter_axes[-1].get_position().x1
    figure.text(
        (scatter_left + scatter_right) / 2.0,
        scatter_axes[0].get_position().y0 - 0.055,
        r"Requested $\Omega_m$",
        ha="center",
        va="center",
    )
    scatter_axes[0].text(
        0.0,
        1.025,
        "(a)",
        transform=scatter_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    transition_axis.text(
        0.0,
        1.025,
        "(b)",
        transform=transition_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    return figure, report


def build_generalization_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Restyle the existing PCA q95 generalization curve without changing values."""

    set_paper_style()
    required = {"arch", "dataset_size", "gen_gl_q95"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"generalization table is missing columns: {missing}")
    figure, axis = plt.subplots(figsize=(FULL_W, 2.35))
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.94)
    for arch, label, color, marker in ARCHITECTURES:
        sub = metrics[metrics["arch"] == arch].copy()
        _require_sizes(sub)
        sub["power"] = np.log2(sub["dataset_size"].astype(float))
        sub = sub.sort_values("power")
        axis.plot(
            sub["power"],
            sub["gen_gl_q95"],
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.25,
            label=label,
        )
    style_training_size_axis(axis)
    axis.axhline(0.5, color="0.50", ls=":", lw=0.8, label="_nolegend_")
    axis.set_ylim(-0.03, 1.04)
    axis.set_xlabel(r"Training images $N_{2D}$")
    axis.set_ylabel("PCA q95 novelty score")
    axis.legend(frameon=False, ncol=3, loc="lower right", handlelength=1.8, columnspacing=1.1)
    return figure


def crop_nearest_training_image(
    image: np.ndarray,
    *,
    top_crop_fraction: float = 0.075,
) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got shape {image.shape}")
    if not 0.0 <= top_crop_fraction < 0.5:
        raise ValueError("top_crop_fraction must be in [0, 0.5)")
    first_row = int(round(image.shape[0] * top_crop_fraction))
    return image[first_row:].copy()


def export_nearest_training_pdf(
    source: Path,
    output: Path,
    *,
    top_crop_fraction: float = 0.075,
) -> tuple[float, float]:
    """Wrap the exact raster audit in a title-free, full-width vector PDF page."""

    set_paper_style()
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"missing exact U-Net-128 nearest-training figure: {source}")
    image = crop_nearest_training_image(
        plt.imread(source),
        top_crop_fraction=top_crop_fraction,
    )
    height = FULL_W * image.shape[0] / image.shape[1]
    figure = plt.figure(figsize=(FULL_W, height))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(image, interpolation="nearest")
    axis.set_axis_off()
    save_figure(figure, output)
    plt.close(figure)
    return FULL_W, height


def build_probe_summary_figure(summary: pd.DataFrame) -> plt.Figure:
    """Show held-out-real VGG+MLP response slope and R2 for all parameters."""

    set_paper_style()
    required = {"parameter", "slope", "r2"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"probe summary is missing columns: {missing}")
    indexed = summary.set_index("parameter")
    missing_parameters = [parameter for parameter in PARAM_ORDER if parameter not in indexed.index]
    if missing_parameters:
        raise ValueError(f"probe summary is missing parameters: {missing_parameters}")
    ordered = indexed.loc[list(PARAM_ORDER)].reset_index()
    x = np.arange(len(PARAM_ORDER))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(PARAM_ORDER)))
    figure, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2))
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.25, top=0.95, wspace=0.24)
    for axis, metric, ylabel in zip(
        axes,
        ("slope", "r2"),
        ("Held-out-real response slope", r"Held-out-real $R^2$"),
    ):
        values = ordered[metric].to_numpy(float)
        axis.axhline(1.0, color="0.38", ls="--", lw=0.9, label="_nolegend_")
        axis.axhline(0.0, color="0.65", lw=0.65, label="_nolegend_")
        axis.plot(x, values, color="0.60", lw=0.75, zorder=1)
        axis.scatter(x, values, c=colors, s=24, marker="o", edgecolor="white", linewidth=0.4, zorder=2)
        for xpos, value in zip(x, values):
            axis.annotate(f"{value:.2f}", (xpos, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
        axis.set_xticks(x, [PARAM_LABELS[parameter] for parameter in PARAM_ORDER])
        axis.set_ylabel(ylabel)
        finite = values[np.isfinite(values)]
        axis.set_ylim(min(-0.15, float(finite.min()) - 0.12), max(1.05, float(finite.max()) + 0.12))
        style_axis(axis)
    axes[0].text(0.01, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(0.01, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    return figure


def save_figure(figure: plt.Figure, output: Path) -> tuple[float, float]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", facecolor="white")
    width, height = figure.get_size_inches()
    return float(width), float(height)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--slopes", type=Path, required=True)
    parser.add_argument("--generalization", type=Path, required=True)
    parser.add_argument("--nearest", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR / "figures")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = {
        "conditional": args.output_dir / "conditional_recovery_transition.pdf",
        "generalization": args.output_dir / "generalization_transition.pdf",
        "nearest": args.output_dir / "nearest_training_unet128.pdf",
        "probe": args.output_dir / "vgg_probe_heldout_real.pdf",
    }
    points = pd.read_csv(args.points)
    slopes = pd.read_csv(args.slopes)
    conditional, report = build_conditional_recovery_figure(points, slopes)
    dimensions = {"conditional": save_figure(conditional, outputs["conditional"])}
    plt.close(conditional)
    generalization = build_generalization_figure(pd.read_csv(args.generalization))
    dimensions["generalization"] = save_figure(generalization, outputs["generalization"])
    plt.close(generalization)
    dimensions["nearest"] = export_nearest_training_pdf(args.nearest, outputs["nearest"])
    probe = build_probe_summary_figure(pd.read_csv(args.probe_summary))
    dimensions["probe"] = save_figure(probe, outputs["probe"])
    plt.close(probe)
    for name, output in outputs.items():
        print(f"{name}: {output} ({dimensions[name][0]:.3f} x {dimensions[name][1]:.3f} in)")
    print(report[["dataset_size", "slope", "slope_ci16", "slope_ci84"]].to_string(index=False))


if __name__ == "__main__":
    main()
