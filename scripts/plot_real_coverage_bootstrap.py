#!/usr/bin/env python
"""Bootstrap uncertainty on the real-map 1-to-1 and coverage curves, and the w = 1 CFG coverage against that band.

Real maps: all 128 z = 0 slices of each held-out simulation 900-931 through the frozen probe
(plot_iaifi_statement_figure.load), per box median and central-p ranges exactly as in plot_probe_bias_vs_scatter.py.
Bootstrap: resample the 32 held-out cosmologies with replacement (B = 10000, seed 0); the slices of a box stay together,
because the box, not the slice, is the independent unit. Each replicate recomputes the coverage curve, covered68,
median bias, slope and intercept of median vs truth. Bands are the 16-84% and 2.5-97.5% of the replicates.
Generated: CFG (Omega_m, sigma_8) models at guidance w = 1, N = 2^6, 2^10, 2^15 (plot_cfg_paper_recovery_figure.draws_for,
64 draws per cosmology). Their curve is compared with the real band, and with a paired bootstrap of
(generated - real) coverage that uses the same resampled cosmologies for both.
Reads saved predictions only. Writes a new timestamped folder; never overwrites.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "paper" / "ai4science_verification"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from paperstyle import FULL_W, set_paper_style, style_axis  # noqa: E402
from plot_cfg_paper_recovery_figure import LEVELS, STYLES, draws_for  # noqa: E402
from plot_iaifi_statement_figure import TEST, load as load_real  # noqa: E402

B, SEED, W = 10000, 0, 1.0
REAL_COLOR = "0.45"


def covered(draws, truth):
    """(levels, boxes) bool: truth inside the central-p range of that box's predictions."""
    lo = np.quantile(draws, (1 - LEVELS) / 2, axis=1)
    hi = np.quantile(draws, (1 + LEVELS) / 2, axis=1)
    return (lo <= truth) & (truth <= hi)


def fit(x, y):
    """Least-squares slope and intercept along the last axis."""
    xm, ym = x.mean(-1, keepdims=True), y.mean(-1, keepdims=True)
    slope = ((x - xm) * (y - ym)).sum(-1) / ((x - xm) ** 2).sum(-1)
    return slope, ym[..., 0] - slope * xm[..., 0]


def q(a, axis=0):
    return np.quantile(a, [0.025, 0.16, 0.84, 0.975], axis=axis)


def main():
    out = ROOT / "results" / f"real_coverage_bootstrap_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    truth, d = load_real()
    real = d["real"]
    assert real.shape == (32, 128), real.shape
    idx = np.random.default_rng(SEED).integers(0, 32, size=(B, 32))

    ind = covered(real, truth)                                   # (L, 32)
    cov = ind.mean(1)
    cov_b = ind[:, idx].mean(-1)                                 # (L, B)
    band = q(cov_b, axis=1)                                      # (4, L)
    pd.DataFrame(dict(nominal=LEVELS, real=cov, lo95=band[0], lo68=band[1], hi68=band[2], hi95=band[3])).to_csv(
        out / "real_coverage_bootstrap_curve.csv", index=False)

    med = np.median(real, 1)
    q16, q84 = np.quantile(real, [0.16, 0.84], axis=1)
    bias = med - truth
    slope, icpt = fit(truth, med)
    sb, ib = fit(truth[idx], med[idx])
    stats = {"covered68 (of 32)": (ind[np.isclose(LEVELS, 0.68)][0].sum(), 32 * cov_b[np.isclose(LEVELS, 0.68)][0]),
             "covered95 (of 32)": (ind[np.isclose(LEVELS, 0.95)][0].sum(), 32 * cov_b[np.isclose(LEVELS, 0.95)][0]),
             "median bias": (np.median(bias), np.median(bias[idx], 1)),
             "median |bias|": (np.median(np.abs(bias)), np.median(np.abs(bias)[idx], 1)),
             "slope of median vs truth": (slope, sb), "intercept": (icpt, ib)}
    rows = [dict(stat=k, value=float(v), ci95_lo=c[0], ci68_lo=c[1], ci68_hi=c[2], ci95_hi=c[3])
            for k, (v, bs) in stats.items() for c in [q(bs)]]
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "real_bootstrap_summary.csv", index=False)

    # generated, CFG w = 1; paired bootstrap of (generated - real) coverage with the same resampled cosmologies
    gen = {p: draws_for(2**p, W, truth) for p in STYLES}
    prow, gcurves = [], {}
    for p, g in gen.items():
        gi = covered(g, truth)
        gc = gi.mean(1); gcurves[p] = gc
        diff_b = gi[:, idx].mean(-1) - cov_b
        dq = q(diff_b, axis=1)
        for lv in (0.68, 0.95):
            k = np.isclose(LEVELS, lv)
            prow.append(dict(N=2**p, level=lv, generated=float(gc[k][0]), real=float(cov[k][0]),
                             real_band68=f"[{band[1][k][0]:.3f}, {band[2][k][0]:.3f}]",
                             real_band95=f"[{band[0][k][0]:.3f}, {band[3][k][0]:.3f}]",
                             diff=float(gc[k][0] - cov[k][0]), diff_ci95_lo=float(dq[0][k][0]),
                             diff_ci95_hi=float(dq[3][k][0]), generated_minus_real_ci95_excludes_0=bool(dq[3][k][0] < 0 or dq[0][k][0] > 0)))
        inside = (gc >= band[0]) & (gc <= band[3])
        prow.append(dict(N=2**p, level="all", generated=np.nan, real=np.nan,
                         real_band68="", real_band95=f"curve inside real 95% band at {inside.sum()}/{len(LEVELS)} levels",
                         diff=np.nan, diff_ci95_lo=np.nan, diff_ci95_hi=np.nan, generated_minus_real_ci95_excludes_0=None))
    paired = pd.DataFrame(prow)
    paired.to_csv(out / "cfg_w1_vs_real_paired_bootstrap.csv", index=False)
    pd.DataFrame({"nominal": LEVELS, "real": cov, **{f"N{2**p}_w1": gcurves[p] for p in STYLES}}).to_csv(
        out / "coverage_curves.csv", index=False)

    set_paper_style()
    # Figure 1: real maps only, with bootstrap uncertainty
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.2), gridspec_kw=dict(wspace=0.32, left=0.10, right=0.97,
                                                                             bottom=0.17, top=0.9))
    ax = axes[0]
    lim = (0.08, 0.52)
    xs = np.linspace(*lim, 50)
    lines = sb[:, None] * xs + ib[:, None]
    lq = q(lines)
    ax.fill_between(xs, lq[0], lq[3], color=REAL_COLOR, alpha=0.15, lw=0, label="bootstrap 95% of the fit")
    ax.fill_between(xs, lq[1], lq[2], color=REAL_COLOR, alpha=0.25, lw=0, label="bootstrap 68% of the fit")
    ax.plot(xs, slope * xs + icpt, color=REAL_COLOR, lw=1.0, label=f"fit, slope {slope:.2f}")
    ax.plot(lim, lim, color="0.15", ls="--", lw=0.9, label="recovered = true")
    inside = ind[np.isclose(LEVELS, 0.68)][0]
    for m, face in ((inside, "0.25"), (~inside, "white")):
        ax.errorbar(truth[m], med[m], yerr=[med[m] - q16[m], q84[m] - med[m]], fmt="o", ms=3, mfc=face, mec="0.25",
                    mew=0.7, ecolor="0.25", elinewidth=0.7, capsize=1.2, zorder=3)
    s = summ.set_index("stat")
    c68 = s.loc["covered68 (of 32)"]
    ax.text(0.10, 0.50, f"{int(c68.value)}/32 bars contain truth\n(bootstrap 95%: {c68.ci95_lo:.0f}–{c68.ci95_hi:.0f})\n"
            f"slope {slope:.2f} [{s.loc['slope of median vs truth'].ci95_lo:.2f}, {s.loc['slope of median vs truth'].ci95_hi:.2f}]",
            fontsize=7.5, va="top")
    ax.set(xlim=lim, ylim=lim, xlabel=r"True $\Omega_m$", ylabel=r"Recovered $\Omega_m$ (128 real slices)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    style_axis(ax)
    ax = axes[1]
    ax.fill_between(LEVELS, band[0], band[3], color=REAL_COLOR, alpha=0.18, lw=0, label="bootstrap 95%")
    ax.fill_between(LEVELS, band[1], band[2], color=REAL_COLOR, alpha=0.32, lw=0, label="bootstrap 68%")
    ax.plot(LEVELS, cov, color="0.2", lw=1.4, label="real maps (this realization)")
    ax.plot((0, 1), (0, 1), color="0.15", ls="--", lw=0.9, label="coverage = nominal")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Nominal coverage p", ylabel="Empirical coverage (of 32 boxes)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    style_axis(ax)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"real_map_bootstrap.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    # Figure 2: paper style, CFG w = 1 at low / intermediate / large N against the real-map bootstrap band
    fig = plt.figure(figsize=(FULL_W, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.06), left=0.10, right=0.96, bottom=0.18, top=0.84, wspace=0.32)
    rax, cax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    rax.plot(lim, lim, color="0.15", ls="--", lw=0.9, zorder=0)
    handles, labels = [], []
    for p, st in STYLES.items():
        g = gen[p]
        gm, g16, g84 = np.median(g, 1), np.quantile(g, 0.16, 1), np.quantile(g, 0.84, 1)
        o = np.argsort(truth)
        h = rax.errorbar(truth[o], gm[o], yerr=np.vstack((np.maximum(gm - g16, 0)[o], np.maximum(g84 - gm, 0)[o])),
                         fmt=st["marker"], ms=3.0, capsize=1.3, elinewidth=0.65, color=st["color"], ecolor=st["color"],
                         markeredgecolor="white", markeredgewidth=0.35, alpha=0.78, zorder=2)
        handles.append(h); labels.append(st["label"])
        cax.plot(LEVELS, gcurves[p], color=st["color"], lw=1.45, zorder=3)
        for lv, mk in ((0.68, "o"), (0.95, "s")):
            cax.scatter([lv], gcurves[p][np.isclose(LEVELS, lv)], s=22, marker=mk, color=st["color"], edgecolor="0.15",
                        linewidth=0.45, zorder=4)
    rax.set(xlim=lim, ylim=lim)
    rax.set_aspect("equal", adjustable="box")
    rax.set_xlabel(r"Requested $\Omega_m$"); rax.set_ylabel(r"Recovered $\Omega_m$")
    style_axis(rax)
    cax.fill_between(LEVELS, band[0], band[3], color=REAL_COLOR, alpha=0.18, lw=0, zorder=1)
    cax.fill_between(LEVELS, band[1], band[2], color=REAL_COLOR, alpha=0.30, lw=0, zorder=1)
    hr, = cax.plot(LEVELS, cov, color=REAL_COLOR, lw=1.3, zorder=2)
    handles.append(hr); labels.append("real maps (band: bootstrap 68%, 95%)")
    cax.plot((0, 1), (0, 1), color="0.15", ls="--", lw=0.9, zorder=0)
    cax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6),
            xlabel="Nominal coverage", ylabel="Empirical coverage")
    style_axis(cax)
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=2,
               handlelength=1.6, columnspacing=1.1, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"conditional_recovery_cfg_g1_realband.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    print(summ.round(4).to_string(index=False))
    print(paired.round(3).to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
