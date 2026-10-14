#!/usr/bin/env python
"""CFG (Omega_m, sigma_8) conditional models by guidance weight w: recovery, P(k) and coverage figures.

Ledger nf_cond_omsigc_cfg. Reads only evaluate_run.py outputs under
results/nf_cond_omsigc_cfg_200k/evaluate_run_v1/<run>/ (per-sample probe predictions, <run>__g<w>_pk.npz, per_run.csv)
plus the real-map probe reference used by plot_iaifi_statement_figure.py.
  1. omega_m_recovery_grid: rows N, columns w; median and 16-84% of the 64 recovered Omega_m per held-out cosmology,
     filled = truth inside the 68% interval.
  2. pk_ratio_by_w: one panel per N; P_gen/P_real per k bin (exp of the mean over 32 cosmologies of log ratio),
     grey band = real split-half ratio (16-84% over cosmologies), i.e. the noise floor.
  3. coverage_by_w: Omega_m interval calibration (same curves as plot_cfg_coverage_by_w.py).
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

from plot_cfg_coverage_by_w import COLOR, EVAL, MANIFEST, REAL, W_VALUES, label, load_cfg
from plot_iaifi_statement_figure import INK, LEVELS, MUTED, load, style, summarize

ROOT = Path(__file__).resolve().parents[1]
W_REAL = 0.0931
GRID_W = (0, 0.5, 0.75, 1, 1.5, 2)


def nlabel(n: int) -> str:
    return f"$N = {n:,}$".replace(",", "{,}")


def recovery_grid(rows, truth, draws, real, out):
    fig, axes = plt.subplots(len(rows), len(GRID_W) + 1, figsize=(7.2, 1.2 * len(rows) + 0.4), sharex=True,
                             sharey=True, gridspec_kw=dict(wspace=0.06, hspace=0.12, left=0.075, right=0.99,
                                                           bottom=0.07, top=0.95))
    lim, ticks = (0.08, 0.52), [0.1, 0.3, 0.5]
    for i, r in enumerate(rows):
        n = r["dataset_size"]
        for j, w in enumerate(("real",) + GRID_W):
            ax = axes[i, j]
            s = real if w == "real" else summarize(truth, draws[n, w])
            c = REAL if w == "real" else COLOR[w]
            ax.plot(lim, lim, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
            for mask, face in ((s["covered"], c), (~s["covered"], "white")):
                ax.errorbar(truth[mask], s["med"][mask], yerr=[s["med"][mask] - s["q16"][mask],
                                                              s["q84"][mask] - s["med"][mask]],
                            fmt="o", ms=2.2, mfc=face, mec=c, mew=0.6, ecolor=c, elinewidth=0.6, capsize=0, zorder=3)
            bias = np.median(s["med"] - truth) / W_REAL
            ax.text(0.04, 0.96, f"{int(s['covered'].sum())}/32\n{bias:+.2f} W", transform=ax.transAxes, ha="left",
                    va="top", fontsize=5.6, color=INK, linespacing=1.1)
            ax.set(xlim=lim, ylim=lim, xticks=ticks, yticks=ticks, aspect="equal")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title("Real maps" if w == "real" else f"$w = {w:g}$", fontsize=7.5, pad=3)
            if j == 0:
                ax.set_ylabel(nlabel(n) + "\nrecovered", fontsize=6.8)
            if i == len(rows) - 1:
                ax.set_xlabel(r"true $\Omega_\mathrm{m}$", fontsize=6.8)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"omega_m_recovery_grid.{ext}")
    plt.close(fig)


def pk_figure(rows, out):
    fig, axes = plt.subplots(1, len(rows), figsize=(7.2, 2.1), sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.07, right=0.86, bottom=0.2, top=0.88))
    table = []
    for ax, r in zip(axes, rows):
        run, n = r["run_name"], r["dataset_size"]
        ref = None
        for w in W_VALUES:
            z = np.load(EVAL / run / f"{run}__{label(w)}_pk.npz")
            kc, ratio = z["kc"], np.exp(np.log(z["ratios"]).mean(0))
            ref = z["split_half"] if ref is None else ref
            ax.plot(kc, ratio, color=COLOR[w], lw=1.5 if w == 1 else 1.0, zorder=4 if w == 1 else 3,
                    label=f"$w = {w:g}$")
            table += [dict(N=n, w=w, kc=float(k), pk_ratio=float(v)) for k, v in zip(kc, ratio)]
        lo, hi = np.quantile(ref, [0.16, 0.84], axis=0)
        ax.fill_between(kc, lo, hi, color=REAL, alpha=0.35, lw=0, zorder=2, label="real split-half")
        ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log")
        ax.set(ylim=(0.3, 1.6), title=nlabel(n), xlabel="$k$ (grid units)")
        ax.title.set_fontsize(8)
    axes[0].set_ylabel(r"$P_\mathrm{gen}/P_\mathrm{real}$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.865, 0.54), fontsize=6.6, frameon=False, handlelength=1.4,
               labelspacing=0.32)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pk_ratio_by_w.{ext}")
    plt.close(fig)
    pd.DataFrame(table).to_csv(out / "pk_ratio_by_w.csv", index=False)


def coverage_figure(rows, truth, draws, real, out):
    fig, axes = plt.subplots(1, len(rows), figsize=(7.2, 1.95), sharey=True,
                             gridspec_kw=dict(wspace=0.1, left=0.065, right=0.86, bottom=0.22, top=0.86))
    for ax, r in zip(axes, rows):
        n = r["dataset_size"]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.plot(LEVELS, real["curve"], color=REAL, lw=2.4, alpha=0.55, zorder=2, label="Real maps")
        for w in W_VALUES:
            ax.plot(LEVELS, summarize(truth, draws[n, w])["curve"], color=COLOR[w], lw=1.5 if w == 1 else 1.0,
                    zorder=4 if w == 1 else 3, label=f"$w = {w:g}$")
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal",
               xticklabels=["0", "0.5", "1"], yticklabels=["0", "0.5", "1"], title=nlabel(n),
               xlabel="Nominal interval")
        ax.title.set_fontsize(8)
    axes[0].set_ylabel("Fraction containing truth")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.865, 0.54), fontsize=6.6, frameon=False, handlelength=1.4,
               labelspacing=0.32)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"coverage_by_w.{ext}")
    plt.close(fig)


def main():
    rows = sorted(json.loads(MANIFEST.read_text()), key=lambda r: r["dataset_size"])
    missing = [r["run_name"] for r in rows if not (EVAL / r["run_name"] / "per_run.csv").exists()]
    if missing:
        raise SystemExit(f"evaluation not finished for {missing}")
    truth, real_draws = load()
    real = summarize(truth, real_draws["real"])
    draws = {(r["dataset_size"], w): load_cfg(r["run_name"], w, truth) for r in rows for w in W_VALUES}
    out = ROOT / "results" / f"cfg_results_by_w_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    style()
    recovery_grid(rows, truth, draws, real, out)
    pk_figure(rows, out)
    coverage_figure(rows, truth, draws, real, out)
    summary = []
    for r in rows:
        pr = pd.read_csv(EVAL / r["run_name"] / "per_run.csv")
        for w in W_VALUES:
            x = pr[pr.run_name == f"{r['run_name']}__{label(w)}"].iloc[0]
            summary.append(dict(N=r["dataset_size"], w=w, bias_over_W=x.Omega_m_median_bias / W_REAL,
                                covered68=int(x.Omega_m_covered68), width68=x.Omega_m_median_width68,
                                coverage95=x.Omega_m_coverage95, slope=x.Omega_m_recovery_slope,
                                pk_large=x.pk_large, pk_mid=x.pk_mid, pk_small=x.pk_small,
                                nearest_train_cos=x.nearest_train_cosine_median,
                                draw_to_draw_cos=x.draw_to_draw_cosine_median,
                                sigma8_bias=x.sigma_8_median_bias, sigma8_covered68=int(x.sigma_8_covered68)))
    s = pd.DataFrame(summary)
    s.to_csv(out / "summary_by_N_w.csv", index=False)
    (out / "real_reference.json").write_text(json.dumps(dict(
        covered68=int(real["covered"].sum()), n=32, width68=float(np.median(real["q84"] - real["q16"])),
        coverage95=float(real["curve"][np.isclose(LEVELS, 0.95)][0]),
        median_bias_over_W=float(np.median(real["med"] - truth) / W_REAL)), indent=2) + "\n")
    print(s.round(3).to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
