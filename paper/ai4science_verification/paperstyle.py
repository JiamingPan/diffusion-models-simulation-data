"""Shared Matplotlib style for the paper figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


FULL_W = 6.75
INK = "#3f3f3f"


def set_paper_style() -> None:
    """Apply vector-safe typography and quiet axis styling."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIX Two Text", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "savefig.bbox": None,
        }
    )


def style_axis(axis) -> None:
    """Enforce paper axes even if an earlier notebook cell changed rcParams."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(axis="both", direction="out", colors=INK, width=0.65, length=3.0)
