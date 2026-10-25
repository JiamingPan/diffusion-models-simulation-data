#!/usr/bin/env python
"""CFG w = 1 recovery and coverage figure, restyled per Nick Kern (2026-10-24).

Left: recovered vs requested Omega_m for the CFG (Omega_m, sigma_8) models at w = 1, N = 2^6, 2^10, 2^15 (median and
16-84% of 64 draws per held-out cosmology), with the probe on real held-out maps shown as a light grey shaded region
(16-84% of the 128 real z = 0 slices per simulation, cosmologies sorted by true Omega_m) as the ground truth.
Right: coverage curves of the three models, each with its own bootstrap 68% band (10,000 resamples of the 32 held-out
cosmologies with replacement, the draws of a cosmology kept together). No real-map curve, no 68%/95% markers.
Inputs as plot_real_coverage_bootstrap.py. Reads saved predictions only. Writes a new timestamped folder.
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
from plot_iaifi_statement_figure import load as load_real  # noqa: E402
from plot_real_coverage_bootstrap import covered  # noqa: E402

B, SEED, W = 10000, 0, 1.0


def main():
    out = ROOT / "results" / f"cfg_w1_bootstrap_bands_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    truth, d = load_real()
    real = d["real"]
    idx = np.random.default_rng(SEED).integers(0, 32, size=(B, 32))
    gen = {p: draws_for(2**p, W, truth) for p in STYLES}

    rows = {"nominal": LEVELS}
    summ = []
    for p, g in gen.items():
        ind = covered(g, truth)
        cb = ind[:, idx].mean(-1)
        rows[f"N{2**p}"] = ind.mean(1)
        rows[f"N{2**p}_lo68"], rows[f"N{2**p}_hi68"] = np.quantile(cb, [0.16, 0.84], axis=1)
        for lv in (0.68, 0.95):
            k = np.isclose(LEVELS, lv)
            summ.append(dict(N=2**p, level=lv, covered=int(round(32 * ind.mean(1)[k][0])),
                             boot68_lo=32 * float(np.quantile(cb[k][0], 0.16)), boot68_hi=32 * float(np.quantile(cb[k][0], 0.84)),
                             boot95_lo=32 * float(np.quantile(cb[k][0], 0.025)), boot95_hi=32 * float(np.quantile(cb[k][0], 0.975))))
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "coverage_curves_bootstrap.csv", index=False)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "covered_counts_bootstrap.csv", index=False)
    o = np.argsort(truth)
    r16, r84 = np.quantile(real, [0.16, 0.84], axis=1)
    pd.DataFrame(dict(truth=truth[o], real_q16=r16[o], real_q84=r84[o])).to_csv(out / "real_probe_band.csv", index=False)

    set_paper_style()
    fig = plt.figure(figsize=(FULL_W, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.06), left=0.10, right=0.96, bottom=0.18, top=0.84, wspace=0.32)
    rax, cax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    lim = (0.08, 0.52)
    hreal = rax.fill_between(truth[o], r16[o], r84[o], color="0.82", lw=0, zorder=0)
    rax.plot(lim, lim, color="0.15", ls="--", lw=0.9, zorder=1)
    handles, labels = [hreal], ["probe on real maps (16–84%)"]
    for p, st in STYLES.items():
        g = gen[p]
        gm, g16, g84 = np.median(g, 1), np.quantile(g, 0.16, 1), np.quantile(g, 0.84, 1)
        h = rax.errorbar(truth[o], gm[o], yerr=np.vstack((np.maximum(gm - g16, 0)[o], np.maximum(g84 - gm, 0)[o])),
                         fmt=st["marker"], ms=3.0, capsize=1.3, elinewidth=0.65, color=st["color"], ecolor=st["color"],
                         markeredgecolor="white", markeredgewidth=0.35, alpha=0.78, zorder=2)
        handles.append(h); labels.append(st["label"])
        n = f"N{2**p}"
        cax.fill_between(LEVELS, curves[f"{n}_lo68"], curves[f"{n}_hi68"], color=st["color"], alpha=0.18, lw=0, zorder=1)
        cax.plot(LEVELS, curves[n], color=st["color"], lw=1.45, zorder=2)
    rax.set(xlim=lim, ylim=lim)
    rax.set_aspect("equal", adjustable="box")
    rax.set_xlabel(r"Requested $\Omega_m$"); rax.set_ylabel(r"Recovered $\Omega_m$")
    style_axis(rax)
    cax.plot((0, 1), (0, 1), color="0.15", ls="--", lw=0.9, zorder=0)
    cax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6),
            xlabel="Nominal coverage", ylabel="Empirical coverage")
    style_axis(cax)
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=2,
               handlelength=1.6, columnspacing=1.1, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"conditional_recovery_cfg_g1_bootbands.{ext}", dpi=300, facecolor="white")
    print(summ.round(1).to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
