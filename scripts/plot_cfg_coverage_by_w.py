#!/usr/bin/env python
"""Omega_m interval calibration of the CFG (Omega_m, sigma_8) conditional models, one panel per N, one curve per w.

Ledger nf_cond_omsigc_cfg. For each N and guidance weight w: empirical fraction of the 32 held-out cosmologies
(sims 900-931) whose central interval of the 64 recovered Omega_m draws contains the truth, against the nominal
interval. w = 1 is plain conditioning (the separate no-guidance set is the same up to float rounding and is not
drawn). Grey: real held-out maps through the same probe (128 slices each), as in plot_iaifi_statement_figure.py.
Data: results/nf_cond_omsigc_cfg_200k/evaluate_run_v1/<run>/<run>__g<w>_per_sample_predictions.csv.
Writes a new timestamped folder; never overwrites.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_iaifi_statement_figure import INK, LEVELS, MUTED, TEST, load, style, summarize

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results/nf_cond_omsigc_cfg_200k/evaluate_run_v1"
MANIFEST = ROOT / "local/nf_cond_omsigc_cfg_200k/manifest.json"
W_VALUES = (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3)
# Diverging around w = 1: blue ramp for w < 1 (dark = w far below 1), neutral ink at w = 1, orange ramp for w > 1.
COLOR = {0: "#0d3b7a", 0.25: "#1f5fb4", 0.5: "#4f8fd8", 0.75: "#94bdea", 1: INK,
         1.5: "#f2a36b", 2: "#e0702e", 3: "#a8430f"}
REAL = "#9a9893"


def label(w: float) -> str:
    return "g" + f"{w:g}".replace(".", "p")


def load_cfg(run: str, w: float, truth: np.ndarray) -> np.ndarray:
    name = f"{run}__{label(w)}"
    d = pd.read_csv(EVAL / run / f"{name}_per_sample_predictions.csv")
    d = d[(d.run_name == name) & (d.parameter == "Omega_m")]
    d = d.set_index(["heldout_sim", "seed_index"]).sort_index()
    assert np.allclose(d.groupby(level=0).theta_in.first().loc[TEST], truth, atol=1e-5), name
    return d.theta_rec.loc[TEST].to_numpy().reshape(32, -1)


def main():
    rows = sorted(json.loads(MANIFEST.read_text()), key=lambda r: r["dataset_size"])
    truth, real_draws = load()
    real = summarize(truth, real_draws["real"])
    out = ROOT / "results" / f"cfg_coverage_by_w_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    style()
    fig, axes = plt.subplots(1, len(rows), figsize=(7.2, 1.95), sharey=True,
                             gridspec_kw=dict(wspace=0.1, left=0.065, right=0.86, bottom=0.22, top=0.86))
    table, curves = [], []
    for ax, r in zip(axes, rows):
        n = r["dataset_size"]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.plot(LEVELS, real["curve"], color=REAL, lw=2.4, alpha=0.55, solid_capstyle="round", zorder=2,
                label="Real maps")
        for w in W_VALUES:
            s = summarize(truth, load_cfg(r["run_name"], w, truth))
            ax.plot(LEVELS, s["curve"], color=COLOR[w], lw=1.1 if w != 1 else 1.5, solid_capstyle="round",
                    zorder=4 if w == 1 else 3, label=f"$w = {w:g}$")
            table.append(dict(N=n, w=w, covered68=int(s["covered"].sum()), n=32,
                              coverage95=float(s["curve"][np.isclose(LEVELS, 0.95)][0]),
                              median_bias=float(np.median(s["med"] - truth)),
                              median_width68=float(np.median(s["q84"] - s["q16"]))))
            curves += [dict(N=n, w=w, nominal=float(l), empirical=float(c)) for l, c in zip(LEVELS, s["curve"])]
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal",
               xticklabels=["0", "0.5", "1"], yticklabels=["0", "0.5", "1"])
        ax.set_title(f"$N = {n:,}$".replace(",", "{,}"), pad=4)
        ax.set_xlabel("Nominal interval")
    axes[0].set_ylabel("Fraction containing truth")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.865, 0.54), fontsize=6.6, frameon=False,
               handlelength=1.4, labelspacing=0.32, title=r"$\Omega_\mathrm{m}$ coverage", title_fontsize=6.8)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cfg_omega_m_coverage_by_w.{ext}")
    pd.DataFrame(table).to_csv(out / "summary_by_N_w.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "coverage_curves.csv", index=False)
    (out / "real_reference.json").write_text(json.dumps(dict(
        covered68=int(real["covered"].sum()), n=32, n_draws=real["n_draws"],
        coverage95=float(real["curve"][np.isclose(LEVELS, 0.95)][0])), indent=2) + "\n")
    print(pd.DataFrame(table).round(3).to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
