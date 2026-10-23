#!/usr/bin/env python
"""1-to-1 Omega_m recovery for Nick's fine-tune controls, labelled by total training updates.

Every model was first trained for 200k updates (the "trained model"). Fine-tunes resume that checkpoint (weights,
optimizer and LR schedule) for +25k / +50k / +100k more updates, i.e. 225k / 250k / 300k in total. Same inputs and
helpers as scripts/plot_ftctl_controls.py; reads saved predictions only. Writes a new timestamped folder.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_ftctl_controls import ARMS, COLOR, LABEL, STEPS, run_info
from plot_iaifi_statement_figure import INK, MUTED, load as load_real, style, summarize
from plot_nick_action_items import REAL, W_REAL, draws_for

ROOT = Path(__file__).resolve().parents[1]
COLTITLE = {"real": "real maps\n(128 slices per box)", 0: "trained model\n200k updates",
            25: "fine-tuned\n+25k = 225k total", 50: "fine-tuned\n+50k = 250k total", 100: "fine-tuned\n+100k = 300k total"}


def main():
    out = ROOT / "results" / f"nick_asks_recovery_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    style()
    truth, d = load_real()
    real = summarize(truth, d["real"])
    rows = [(n, a) for n in ARMS for a in ARMS[n]]
    fig, axes = plt.subplots(len(rows), 5, figsize=(7.2, 7.6), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06, hspace=0.12, left=0.15, right=0.99, bottom=0.06, top=0.94))
    lim, ticks = (0.08, 0.52), [0.1, 0.3, 0.5]
    for i, (n, arm) in enumerate(rows):
        for j, key in enumerate(("real",) + STEPS):
            ax = axes[i, j]
            ax.set(xlim=lim, ylim=lim, xticks=ticks, yticks=ticks, aspect="equal")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(COLTITLE[key], fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(f"N = {n:,}\n{LABEL[arm]}\nrecovered " + r"$\Omega_\mathrm{m}$", fontsize=6.4)
            if i == len(rows) - 1:
                ax.set_xlabel(r"true $\Omega_\mathrm{m}$", fontsize=6.8)
            if key == "real":
                s, c = real, REAL
            else:
                r = run_info(n, arm, key)
                s, c = summarize(truth, draws_for(r["pred"], r["run"], truth)), (MUTED if key == 0 else COLOR[arm])
            ax.plot(lim, lim, color=MUTED, lw=0.6, ls=(0, (4, 3)))
            for m, face in ((s["covered"], c), (~s["covered"], "white")):
                ax.errorbar(truth[m], s["med"][m], yerr=[s["med"][m] - s["q16"][m], s["q84"][m] - s["med"][m]], fmt="o",
                            ms=2.2, mfc=face, mec=c, mew=0.6, ecolor=c, elinewidth=0.6, capsize=0)
            ax.text(0.04, 0.96, f"{int(s['covered'].sum())}/32 covered\nbias {np.median(s['med'] - truth) / W_REAL:+.2f} W",
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=INK, linespacing=1.1)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"nick_asks_recovery_grid.{ext}", dpi=300)
    print("Wrote", out)


if __name__ == "__main__":
    main()
