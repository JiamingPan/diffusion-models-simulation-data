#!/usr/bin/env python
"""Paper Fig. 2 style (1-to-1 recovery + coverage curve) for the fine-tune controls.

Same construction and style as results/conditional_recovery_real_reference_*/ (paper Fig. 2 with the real-map curve):
left, recovered vs requested Omega_m (point = median of the 64 draws at each held-out cosmology, bar = 16-84%);
right, empirical coverage(p) = fraction of the 32 cosmologies whose truth lies in the central p range of the draws,
with the real held-out maps (128 slices per box) as the grey reference curve.
  by_arm_<N>_<arm>.png : one model size and arm; series = 200k trained model and its 225k / 250k / 300k fine-tunes.
  by_n_<arm>_300k.png  : one arm at 300k total updates; series = N = 1,024 and N = 4,096.
  row_n<N>_<arm>.png   : one model size and arm; a separate true-vs-observed 1-to-1 panel per checkpoint (filled =
                         16-84% bar contains the truth, open = misses), then the four coverage curves.
Arms and checkpoints from scripts/plot_ftctl_controls.py (run_info). Reads saved per-sample predictions only.
Checks the 68% and 95% counts against each run's evaluate_run.py per_run.csv. New timestamped folder; never overwrites.
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

from paperstyle import FULL_W, set_paper_style, style_axis
from plot_ftctl_controls import ARMS, LABEL, STEPS, run_info
from plot_iaifi_statement_figure import load as load_real
from plot_nf_conditional_bias_paper_figures import COVERAGE_CURVE_LEVELS, COVERAGE_LEVELS, COVERAGE_MARKERS
from plot_nick_action_items import W_REAL, draws_for

ROOT = Path(__file__).resolve().parents[1]
LEVELS = np.asarray(COVERAGE_CURVE_LEVELS, dtype=float)
REAL_COLOR = "0.55"
STEP_STYLE = {0: dict(color="#4D4D4D", marker="o", label="200k, trained model"),
              25: dict(color="#8FB8E8", marker="D", label="225k total"),
              50: dict(color="#2A78D6", marker="s", label="250k total"),
              100: dict(color="#0B2E5C", marker="^", label="300k total")}
N_STYLE = {1024: dict(color="#E67E22", marker="D", label="N = 1,024"),
           4096: dict(color="#1B9E77", marker="s", label="N = 4,096")}


def coverage(y, truth):
    lo = np.quantile(y, (1 - LEVELS) / 2, axis=1)
    hi = np.quantile(y, (1 + LEVELS) / 2, axis=1)
    return ((lo <= truth) & (truth <= hi)).mean(axis=1)


def count(cov, p):
    return int(round(float(cov[np.isclose(LEVELS, p)][0]) * 32))


def pair(series, truth, real, title, path):
    """series: list of (style dict, draws (32, k)). Left 1-to-1, right coverage; same look as paper Fig. 2."""
    many = len(series) > 2
    fig = plt.figure(figsize=(FULL_W, 3.6 if many else 3.35))
    g = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.06), left=0.10, right=0.96, bottom=0.17, top=0.75 if many else 0.80,
                         wspace=0.32)
    rax, cax = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1])
    lim = (0.06, 0.53)
    rax.plot(lim, lim, color="0.15", ls="--", lw=0.9, zorder=0)
    cax.plot((0, 1), (0, 1), color="0.15", ls="--", lw=0.9, zorder=0)
    cax.plot(LEVELS, real, color=REAL_COLOR, lw=1.3, zorder=1.5)
    for p in COVERAGE_LEVELS:
        cax.scatter(p, real[np.isclose(LEVELS, p)], s=22, marker=COVERAGE_MARKERS[p], color=REAL_COLOR,
                    edgecolor="0.15", linewidth=0.45, zorder=1.6)
    cax.text(0.19, 0.345, "real maps", color="0.35", fontsize=8.5, ha="center")
    handles, labels = [], []
    offsets = np.linspace(-0.006, 0.006, len(series)) if len(series) > 1 else [0.0]
    for (st, y), dx in zip(series, offsets):
        med = np.median(y, axis=1)
        q16, q84 = np.quantile(y, [0.16, 0.84], axis=1)
        h = rax.errorbar(truth + dx, med, yerr=[np.maximum(med - q16, 0), np.maximum(q84 - med, 0)], fmt=st["marker"],
                         ms=3.0, capsize=1.3, elinewidth=0.65, color=st["color"], ecolor=st["color"],
                         markeredgecolor="white", markeredgewidth=0.35, alpha=0.78, zorder=2)
        c = coverage(y, truth)
        cax.plot(LEVELS, c, color=st["color"], lw=1.45, zorder=2)
        for p in COVERAGE_LEVELS:
            cax.scatter(p, c[np.isclose(LEVELS, p)], s=22, marker=COVERAGE_MARKERS[p], color=st["color"],
                        edgecolor="0.15", linewidth=0.45, zorder=3)
        handles.append(h)
        labels.append(f"{st['label']} ({count(c, .68)}/32, {count(c, .95)}/32)")
    rax.set(xlim=lim, ylim=lim)
    rax.set_aspect("equal", adjustable="box")
    rax.set_xlabel(r"Requested $\Omega_m$", fontsize=10.5)
    rax.set_ylabel(r"Recovered $\Omega_m$", fontsize=10.5)
    cax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
    cax.set_xlabel("Nominal coverage", fontsize=10.5)
    cax.set_ylabel("Empirical coverage", fontsize=10.5)
    for ax in (rax, cax):
        style_axis(ax)
        ax.tick_params(axis="both", labelsize=9.0)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 0.95), ncol=2 if many else len(series),
               handlelength=1.4, columnspacing=1.2, fontsize=8.3)
    fig.savefig(path.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=200, facecolor="white")
    plt.close(fig)


def row(n, a, draws, truth, real, path):
    """One (N, arm): a separate 1-to-1 panel per checkpoint (no overlay), then the coverage curves."""
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.75), gridspec_kw=dict(wspace=0.28, left=0.055, right=0.995,
                                                                           bottom=0.19, top=0.80))
    lim = (0.06, 0.53)
    cax = axes[4]
    cax.plot((0, 1), (0, 1), color="0.15", ls="--", lw=0.9, zorder=0)
    cax.plot(LEVELS, real, color=REAL_COLOR, lw=1.3, zorder=1.5)
    for p in COVERAGE_LEVELS:
        cax.scatter(p, real[np.isclose(LEVELS, p)], s=18, marker=COVERAGE_MARKERS[p], color=REAL_COLOR,
                    edgecolor="0.15", linewidth=0.45, zorder=1.6)
    cax.text(0.03, 0.47, "real maps", color="0.35", fontsize=8, ha="left")
    for ax, s in zip(axes[:4], STEPS):
        st, y = STEP_STYLE[s], draws[n, a, s]
        med = np.median(y, axis=1)
        q16, q84 = np.quantile(y, [0.16, 0.84], axis=1)
        inside = (q16 <= truth) & (truth <= q84)
        ax.plot(lim, lim, color="0.15", ls="--", lw=0.9, zorder=0)
        for m, face in ((inside, st["color"]), (~inside, "white")):
            ax.errorbar(truth[m], med[m], yerr=[np.maximum(med[m] - q16[m], 0), np.maximum(q84[m] - med[m], 0)],
                        fmt="o", ms=3.4, capsize=1.3, elinewidth=0.7, color=st["color"], ecolor=st["color"],
                        mfc=face, mec=st["color"], mew=0.8, zorder=2)
        c = coverage(y, truth)
        ax.text(0.05, 0.95, f"{count(c, .68)}/32 covered (68%)\n{count(c, .95)}/32 covered (95%)\n"
                f"bias {np.median(med - truth):+.3f} ({np.median(med - truth) / W_REAL:+.2f} W)", transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
                color="0.15", linespacing=1.15)
        ax.set(xlim=lim, ylim=lim, xticks=[0.1, 0.2, 0.3, 0.4, 0.5], yticks=[0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(st["label"], fontsize=9.5, color=st["color"] if s else "0.15", pad=4)
        ax.set_xlabel(r"True $\Omega_m$", fontsize=9.5)
        ax.set_ylabel(r"Observed $\Omega_m$ (probe)" if s == 0 else "", fontsize=9.5)
        cax.plot(LEVELS, c, color=st["color"], lw=1.4, zorder=2)
        for p in COVERAGE_LEVELS:
            cax.scatter(p, c[np.isclose(LEVELS, p)], s=18, marker=COVERAGE_MARKERS[p], color=st["color"],
                        edgecolor="0.15", linewidth=0.45, zorder=3)
    cax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
    cax.set_aspect("equal", adjustable="box")
    cax.set_xlabel("Nominal coverage", fontsize=9.5)
    cax.set_ylabel("Empirical coverage", fontsize=9.5)
    cax.set_title("coverage, all four", fontsize=9.5, pad=4)
    for ax in axes:
        style_axis(ax)
        ax.tick_params(axis="both", labelsize=8)
    fig.suptitle(f"N = {n:,}, {LABEL[a]}", fontsize=11, y=0.985)
    fig.savefig(path.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=200, facecolor="white")
    plt.close(fig)


def main():
    out = ROOT / "results" / f"ftctl_fig2_style_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    set_paper_style()
    truth, d = load_real()
    real = coverage(d["real"], truth)
    draws, rows = {}, []
    for n in ARMS:
        for a in ARMS[n]:
            for s in STEPS:
                r = run_info(n, a, s)
                y = draws_for(r["pred"], r["run"], truth)
                c = coverage(y, truth)
                ref = pd.read_csv(r["per_run"]).set_index("run_name").loc[r["run"]]
                assert count(c, .68) == int(ref.Omega_m_covered68), r["run"]
                assert count(c, .95) == int(round(ref.Omega_m_coverage95 * 32)), r["run"]
                draws[n, a, s] = y
                rows.append(dict(N=n, arm=a, total_updates_k=200 + s, run=r["run"], covered68=count(c, .68),
                                 covered95=count(c, .95), source=str(r["pred"].relative_to(ROOT))))
    pd.DataFrame(rows).to_csv(out / "covered_counts.csv", index=False)
    for n in ARMS:
        for a in ARMS[n]:
            pair([(STEP_STYLE[s], draws[n, a, s]) for s in STEPS], truth, real,
                 f"N = {n:,}, {LABEL[a]}: trained model and fine-tunes", out / f"by_arm_n{n}_{a}")
    for a in ("noaug", "idaug", "d4shift"):
        ns = [n for n in ARMS if a in ARMS[n]]
        pair([(N_STYLE[n], draws[n, a, 100]) for n in ns], truth, real, f"{LABEL[a]}, 300k total updates",
             out / f"by_n_{a}_300k")
    pair([(N_STYLE[n], draws[n, "noaug", 0]) for n in ARMS], truth, real, "trained models, 200k updates",
         out / "by_n_trained_200k")
    for n in ARMS:
        for a in ARMS[n]:
            row(n, a, draws, truth, real, out / f"row_n{n}_{a}")
    print(pd.DataFrame(rows)[["N", "arm", "total_updates_k", "covered68", "covered95"]].to_string(index=False))
    print("real:", count(real, .68), count(real, .95))
    print("Wrote", out)


if __name__ == "__main__":
    main()
