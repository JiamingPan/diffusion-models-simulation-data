"""Paper-ready figures built from audited evaluation tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.ai4science_verification.paperstyle import FULL_W, INK, set_paper_style, style_axis


U128_DATASET_SIZES = tuple(2**exponent for exponent in range(6, 16))


def _complete_u128_curves(curves: list[dict[str, object]]) -> list[dict[str, object]]:
    required = {
        "dataset_size",
        "hist_centers",
        "real_hist",
        "generated_hist",
        "k_bins",
        "pk_ratio",
    }
    selected = [row for row in curves if str(row.get("arch", "u128")) == "u128"]
    selected.sort(key=lambda row: int(row["dataset_size"]))

    observed = tuple(int(row["dataset_size"]) for row in selected)
    if observed != U128_DATASET_SIZES:
        raise ValueError(
            "paper figure requires the complete UNet-128 sweep at "
            f"{list(U128_DATASET_SIZES)}; observed {list(observed)}"
        )

    for row in selected:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(
                f"UNet-128 curve record for N={row['dataset_size']} is missing {missing}"
            )
        centers = np.asarray(row["hist_centers"], dtype=float)
        real_hist = np.asarray(row["real_hist"], dtype=float)
        generated_hist = np.asarray(row["generated_hist"], dtype=float)
        k_bins = np.asarray(row["k_bins"], dtype=float)
        pk_ratio = np.asarray(row["pk_ratio"], dtype=float)
        if any(array.ndim != 1 for array in (centers, real_hist, generated_hist, k_bins, pk_ratio)):
            raise ValueError("paper curve arrays must be one-dimensional")
        if not (len(centers) == len(real_hist) == len(generated_hist)):
            raise ValueError("one-point histogram arrays must have matching lengths")
        if len(k_bins) != len(pk_ratio):
            raise ValueError("power-spectrum arrays must have matching lengths")
        if not np.isfinite(centers).all() or not np.isfinite(k_bins).all():
            raise ValueError("paper curve coordinates contain non-finite values")
        if np.any(real_hist < 0) or np.any(generated_hist < 0):
            raise ValueError("one-point histogram densities must be non-negative")
        if not np.any(np.isfinite(pk_ratio) & (pk_ratio > 0)):
            raise ValueError("power-spectrum ratio has no finite positive values")
    return selected


def _complete_u128_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"arch", "dataset_size", "hist_l1", "pk_log10_mae"}
    missing_columns = sorted(required.difference(metrics.columns))
    if missing_columns:
        raise ValueError(f"metrics table is missing columns: {missing_columns}")

    selected = metrics.loc[
        metrics["arch"].astype(str).eq("u128"),
        ["arch", "dataset_size", "hist_l1", "pk_log10_mae"],
    ].copy()
    selected["dataset_size"] = selected["dataset_size"].astype(int)
    selected = selected.sort_values("dataset_size").reset_index(drop=True)

    observed = tuple(selected["dataset_size"].tolist())
    if observed != U128_DATASET_SIZES:
        raise ValueError(
            "paper figure requires the complete UNet-128 sweep at "
            f"{list(U128_DATASET_SIZES)}; observed {list(observed)}"
        )
    if not np.isfinite(selected[["hist_l1", "pk_log10_mae"]].to_numpy(dtype=float)).all():
        raise ValueError("complete UNet-128 sweep contains non-finite physical metrics")
    return selected


def build_summary_statistics_figure(
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot one- and two-point errors across the complete UNet-128 data sweep."""

    selected = _complete_u128_metrics(metrics)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_W, 2.35),
        constrained_layout=True,
    )
    exponents = np.log2(selected["dataset_size"].to_numpy(dtype=float))
    color = "#00796B"
    panels = (
        ("hist_l1", r"One-point PDF $L_1$ error", "(a) One-point distribution"),
        (
            "pk_log10_mae",
            r"Mean $|\log_{10}(P_{\rm gen}/P_{\rm real})|$",
            "(b) Power spectrum",
        ),
    )

    for axis, (column, ylabel, title) in zip(axes, panels):
        values = selected[column].to_numpy(dtype=float)
        axis.plot(exponents, values, color=INK, linewidth=0.85, zorder=1)
        axis.scatter(
            exponents,
            values,
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=2,
        )
        axis.set_title(title, loc="left", pad=4.0)
        axis.set_ylabel(ylabel)
        axis.set_xlim(5.7, 15.3)
        axis.set_ylim(bottom=0.0)
        axis.set_xticks(
            [6, 8, 10, 12, 14],
            [rf"$2^{{{exponent}}}$" for exponent in [6, 8, 10, 12, 14]],
        )
        axis.grid(False)
        style_axis(axis)

    figure.supxlabel(r"Training images $N_{\rm 2D}$", fontsize=9)
    figure.savefig(
        output,
        format="pdf",
        metadata={"Title": "UNet-128 one- and two-point statistic sweep"},
    )
    return figure, selected


def build_summary_statistics_curve_figure(
    curves: list[dict[str, object]],
    output_path: str | Path,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot the full matched one- and two-point curves for the UNet-128 sweep."""

    selected = _complete_u128_curves(curves)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    histogram_values = np.concatenate(
        [
            np.asarray(row[key], dtype=float)
            for row in selected
            for key in ("real_hist", "generated_hist")
        ]
    )
    histogram_peak = float(np.nanmax(histogram_values))
    histogram_floor = max(histogram_peak * 1.0e-6, np.finfo(float).tiny)
    histogram_log_min = np.log10(histogram_floor)
    histogram_log_max = np.log10(max(histogram_peak, histogram_floor * 10.0))
    histogram_span = max(histogram_log_max - histogram_log_min, 1.0)

    ratio_logs = []
    for row in selected:
        ratio = np.asarray(row["pk_ratio"], dtype=float)
        valid = np.isfinite(ratio) & (ratio > 0)
        ratio_logs.append(np.log10(ratio[valid]))
    ratio_limit = max(float(np.max(np.abs(np.concatenate(ratio_logs)))), 1.0e-3)

    set_paper_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_W, 3.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.05]},
    )
    row_offsets = np.arange(len(selected), dtype=float)
    row_labels = [rf"$2^{{{int(np.log2(size))}}}$" for size in U128_DATASET_SIZES]
    real_color = "#4A4A4A"
    generated_color = "#16857A"

    histogram_axis, spectrum_axis = axes
    for offset, row in zip(row_offsets, selected):
        centers = np.asarray(row["hist_centers"], dtype=float)
        real_hist = np.asarray(row["real_hist"], dtype=float)
        generated_hist = np.asarray(row["generated_hist"], dtype=float)
        real_height = (
            np.log10(np.clip(real_hist, histogram_floor, None)) - histogram_log_min
        ) / histogram_span
        generated_height = (
            np.log10(np.clip(generated_hist, histogram_floor, None)) - histogram_log_min
        ) / histogram_span
        histogram_axis.plot(
            centers,
            offset + 0.72 * real_height,
            color=real_color,
            linewidth=0.75,
            linestyle=(0, (3.0, 2.0)),
            alpha=0.9,
        )
        histogram_axis.plot(
            centers,
            offset + 0.72 * generated_height,
            color=generated_color,
            linewidth=1.05,
        )

        k_bins = np.asarray(row["k_bins"], dtype=float)
        ratio = np.asarray(row["pk_ratio"], dtype=float)
        valid = np.isfinite(k_bins) & np.isfinite(ratio) & (ratio > 0)
        spectrum_axis.hlines(
            offset,
            float(np.nanmin(k_bins)),
            float(np.nanmax(k_bins)),
            color="#A5A5A5",
            linewidth=0.45,
            linestyle=(0, (2.5, 2.0)),
            zorder=0,
        )
        spectrum_axis.plot(
            k_bins[valid],
            offset + 0.36 * np.log10(ratio[valid]) / ratio_limit,
            color=generated_color,
            linewidth=1.05,
            zorder=1,
        )

    histogram_axis.plot(
        [],
        [],
        color=real_color,
        linewidth=0.85,
        linestyle=(0, (3.0, 2.0)),
        label="real",
    )
    histogram_axis.plot([], [], color=generated_color, linewidth=1.1, label="generated")
    histogram_axis.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.2,
        ncol=2,
        columnspacing=0.9,
    )
    histogram_axis.set_title("(a) One-point PDFs (log density)", loc="left", pad=4.0)
    histogram_axis.set_xlabel("Normalized field value")
    histogram_axis.set_ylabel(r"Training images $N_{\rm 2D}$")

    spectrum_axis.set_title("(b) Power-spectrum ratios", loc="left", pad=4.0)
    spectrum_axis.set_xlabel(r"$k$ bin")
    spectrum_axis.set_ylabel(r"Training images $N_{\rm 2D}$")

    for axis in axes:
        axis.set_yticks(row_offsets, row_labels)
        axis.set_ylim(-0.35, len(selected) + 0.35)
        axis.tick_params(axis="y", length=0)
        axis.grid(False)
        style_axis(axis)

    figure.savefig(
        output,
        format="pdf",
        metadata={"Title": "UNet-128 one- and two-point statistic sweep"},
    )
    plotted = pd.DataFrame(
        {
            "arch": ["u128"] * len(selected),
            "dataset_size": [int(row["dataset_size"]) for row in selected],
        }
    )
    return figure, plotted
