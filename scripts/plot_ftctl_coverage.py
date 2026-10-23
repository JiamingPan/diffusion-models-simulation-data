#!/usr/bin/env python
"""Omega_m coverage curves for the fine-tune controls, by arm and total training updates, with the real-map curve.

Same construction as the real-map curve in scripts/plot_probe_bias_vs_scatter.py: for each held-out cosmology i
(sims 900-931) and its 64 generated draws, the central p range [Q_i((1-p)/2), Q_i((1+p)/2)] of the probe readings;
empirical coverage(p) = fraction of the 32 cosmologies whose true Omega_m lies inside. Real maps use their 128 slices.
Arms and checkpoints from scripts/plot_ftctl_controls.py (run_info). Reads saved per-sample predictions only.
Checks the p = 0.68 count against Omega_m_covered68 in each run's evaluate_run.py per_run.csv.
Writes a new timestamped folder; never overwrites.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_ftctl_controls import ARMS, COLOR, LABEL, LS, MARK, STEPS, run_info
from plot_iaifi_statement_figure import INK, MUTED, load as load_real, style
from plot_nick_action_items import REAL, draws_for

ROOT = Path(__file__).resolve().parents[1]
LEVELS = np.unique(np.r_[np.linspace(0, 1, 51), 0.68, 0.95])
STEP_COLOR = {0: "#52514e", 25: "#9cc3ec", 50: "#2a78d6", 100: "#0b2e5c"}
STEP_LABEL = {0: "200k, trained model", 25: "225k total", 50: "250k total", 100: "300k total"}


def coverage(y, truth):
    lo = np.quantile(y, (1 - LEVELS) / 2, axis=1)
    hi = np.quantile(y, (1 + LEVELS) / 2, axis=1)
    return ((lo <= truth) & (truth <= hi)).mean(axis=1)


def at(cov, p):
    return float(cov[np.isclose(LEVELS, p)][0])


def main():
    out = ROOT / "results" / f"ftctl_coverage_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    style()
    truth, d = load_real()
    real = coverage(d["real"], truth)
    curves, rows = {}, []
    for n in ARMS:
        for a in ARMS[n]:
            for s in STEPS:
                r = run_info(n, a, s)
                c = coverage(draws_for(r["pred"], r["run"], truth), truth)
                ref = int(pd.read_csv(r["per_run"]).set_index("run_name").loc[r["run"], "Omega_m_covered68"])
                assert round(at(c, 0.68) * 32) == ref, (r["run"], at(c, 0.68) * 32, ref)
                curves[n, a, s] = c
                rows += [dict(N=n, arm=a, total_updates_k=200 + s, run=r["run"], nominal=lv, empirical=cv,
                              covered=int(round(cv * 32)), source=str(r["pred"].relative_to(ROOT)))
                         for lv, cv in zip(LEVELS, c)]
    rows += [dict(N="real", arm="real", total_updates_k=np.nan, run="real held-out maps (128 slices)", nominal=lv,
                  empirical=cv, covered=int(round(cv * 32)), source="plot_iaifi_statement_figure.load")
             for lv, cv in zip(LEVELS, real)]
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "coverage_curves.csv", index=False)

    def base_axes(ax):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.plot(LEVELS, real, color=REAL, lw=2.4, alpha=0.75, zorder=1, label="real held-out maps")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), aspect="equal", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        ax.tick_params(labelsize=6.5)

    # Figure 1: one panel per (N, arm); one curve per checkpoint
    panels = [(n, a) for n in ARMS for a in ARMS[n]]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), gridspec_kw=dict(wspace=0.12, hspace=0.32, left=0.1,
                                                                          right=0.80, bottom=0.09, top=0.93))
    for ax, (n, a) in zip([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 2]], panels):
        base_axes(ax)
        for s in STEPS:
            c = curves[n, a, s]
            ax.plot(LEVELS, c, color=STEP_COLOR[s], lw=1.3 if s else 1.6, label=STEP_LABEL[s], zorder=3 if s else 2)
            for lv, mk in ((0.68, "o"), (0.95, "s")):
                ax.plot(lv, at(c, lv), mk, ms=3.6, mfc=STEP_COLOR[s], mec="white", mew=0.5, zorder=4)
        c3 = curves[n, a, 100]
        ax.text(0.04, 0.97, f"at 300k total:\n{round(at(c3, .68) * 32)}/32 covered at 68%\n{round(at(c3, .95) * 32)}/32 covered at 95%",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=INK, linespacing=1.15)
        ax.set_title(f"N = {n:,}, {LABEL[a]}", fontsize=7.2, pad=3)
    for ax in axes[:, 1:].ravel():
        ax.tick_params(labelleft=False)
    for ax in (axes[0, 0], axes[0, 2]):
        ax.tick_params(labelbottom=False)
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5, "identity augmentation\nwas run at N = 1,024 only", ha="center", va="center", fontsize=6.8,
                    color=MUTED, transform=axes[1, 1].transAxes)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("fraction of 32 cosmologies\ncontaining truth", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("nominal level p", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=MUTED, lw=0.6, ls=(0, (4, 3))),
          plt.Line2D([], [], ls="", marker="o", ms=3.6, color=INK), plt.Line2D([], [], ls="", marker="s", ms=3.6, color=INK)]
    l += ["coverage = p", "p = 0.68", "p = 0.95"]
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.81, 0.55), fontsize=6.5, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ftctl_coverage_by_arm.{ext}", dpi=300)

    # Figure 2: arms overlaid at 300k total, one panel per N
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.75), gridspec_kw=dict(wspace=0.16, left=0.1, right=0.7, bottom=0.17,
                                                                           top=0.9))
    for ax, n in zip(axes, ARMS):
        base_axes(ax)
        ax.plot(LEVELS, curves[n, "noaug", 0], color=MUTED, lw=1.0, ls=":", label="200k, trained model", zorder=2)
        for a in ARMS[n]:
            c = curves[n, a, 100]
            ax.plot(LEVELS, c, color=COLOR[a], lw=1.4, ls=LS[a], label=f"{LABEL[a]}, 300k", zorder=3)
            for lv in (0.68, 0.95):
                ax.plot(lv, at(c, lv), MARK[a], ms=3.6, mfc=COLOR[a], mec="white", mew=0.5, zorder=4)
        ax.set_title(f"N = {n:,}, all arms at 300k total", fontsize=7.4, pad=3)
        ax.set_xlabel("nominal level p", fontsize=7)
    axes[0].set_ylabel("fraction of 32 cosmologies\ncontaining truth", fontsize=7)
    axes[1].tick_params(labelleft=False)
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=MUTED, lw=0.6, ls=(0, (4, 3)))]
    l += ["coverage = p"]
    h += [plt.Line2D([], [], ls="", marker=MARK[a], ms=3.6, color=COLOR[a]) for a in ARMS[1024]]
    l += [f"{LABEL[a]}: p = 0.68 and 0.95" for a in ARMS[1024]]
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.71, 0.53), fontsize=6.5, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ftctl_coverage_300k.{ext}", dpi=300)

    summ = tab[np.isclose(tab.nominal, 0.68) | np.isclose(tab.nominal, 0.95)].pivot_table(
        index=["N", "arm", "total_updates_k"], columns="nominal", values="covered", dropna=False).dropna(how="all")
    summ.to_csv(out / "coverage_68_95_summary.csv")
    print(summ.to_string())
    print("real 68/95:", round(at(real, .68) * 32), round(at(real, .95) * 32))
    print("Wrote", out)


if __name__ == "__main__":
    main()
