#!/usr/bin/env python
"""Paper-style recovery + coverage figure for the CFG conditional models at one guidance weight w.

Same layout and style as conditional_recovery_transition.pdf (scripts/plot_nf_conditional_bias_paper_figures.py,
build_conditional_coverage_figure): left, median and 16-84% of the frozen-probe Omega_m recovered from 64 generated
maps at each of 32 held-out cosmologies; right, empirical coverage of central intervals vs nominal level. Computes (but does not plot) the
real-map reference (128 real slices per held-out simulation through the same probe; plot_iaifi_statement_figure.load)
as a grey curve, because a per-slice predictive band covers the truth more often than nominal even for real maps.
Data: results/nf_cond_omsigc_cfg_200k/evaluate_run_v1/<run>/<run>__g<w>_per_sample_predictions.csv (ledger
nf_cond_omsigc_cfg). Writes a new timestamped folder; never overwrites.

usage: plot_cfg_paper_recovery_figure.py --w 1.5 [--w 1 ...]
"""
from __future__ import annotations

import argparse
import json
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
PAPER_DIR = ROOT / "paper" / "ai4science_verification"
for p in (ROOT, ROOT / "scripts", PAPER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from paperstyle import FULL_W, set_paper_style, style_axis  # noqa: E402
from plot_iaifi_statement_figure import TEST, load as load_real  # noqa: E402

EVAL = ROOT / "results/nf_cond_omsigc_cfg_200k/evaluate_run_v1"
MANIFEST = ROOT / "local/nf_cond_omsigc_cfg_200k/manifest.json"
LEVELS = np.array(sorted(set(np.linspace(0.0, 1.0, 51).tolist()) | {0.68, 0.95}))
FOCAL = {0.68: "o", 0.95: "s"}
STYLES = {
    6: {"color": "#C23B2A", "marker": "o", "label": r"$N_{2D}=2^{6}$ memorization regime"},
    10: {"color": "#E67E22", "marker": "D", "label": r"$N_{2D}=2^{10}$"},
    15: {"color": "#0072B2", "marker": "s", "label": r"$N_{2D}=2^{15}$ generalization regime"},
}
REAL_COLOR = "0.55"


def wlabel(w: float) -> str:
    return "g" + f"{w:g}".replace(".", "p")


def draws_for(n: int, w: float, truth: np.ndarray) -> np.ndarray:
    run = {r["dataset_size"]: r["run_name"] for r in json.loads(MANIFEST.read_text())}[n]
    name = f"{run}__{wlabel(w)}"
    d = pd.read_csv(EVAL / run / f"{name}_per_sample_predictions.csv")
    d = d[(d.run_name == name) & (d.parameter == "Omega_m")].set_index(["heldout_sim", "seed_index"]).sort_index()
    assert np.allclose(d.groupby(level=0).theta_in.first().loc[TEST], truth, atol=1e-5), name
    out = d.theta_rec.loc[TEST].to_numpy().reshape(len(TEST), -1)
    assert out.shape == (32, 64), out.shape
    return out


def curve(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    lo = np.quantile(draws, (1 - LEVELS) / 2, axis=1)
    hi = np.quantile(draws, (1 + LEVELS) / 2, axis=1)
    return ((lo <= truth) & (truth <= hi)).mean(axis=1)


def figure_for(w: float, truth: np.ndarray, real: np.ndarray):
    set_paper_style()
    fig = plt.figure(figsize=(FULL_W, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.06), left=0.10, right=0.96, bottom=0.18, top=0.84, wspace=0.32)
    rax, cax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    rows, handles, labels, curves = [], [], [], {}
    data = {p: draws_for(2**p, w, truth) for p in STYLES}
    allv = np.concatenate([truth] + [np.quantile(v, [0.16, 0.84], axis=1).ravel() for v in data.values()])
    pad = 0.06 * (allv.max() - allv.min())
    lim = (allv.min() - pad, allv.max() + pad)
    rax.plot(lim, lim, color="0.15", ls="--", lw=0.9, zorder=0)
    for p, st in STYLES.items():
        d = data[p]
        med, q16, q84 = np.median(d, 1), np.quantile(d, 0.16, 1), np.quantile(d, 0.84, 1)
        o = np.argsort(truth)
        h = rax.errorbar(truth[o], med[o], yerr=np.vstack((np.maximum(med - q16, 0)[o], np.maximum(q84 - med, 0)[o])),
                         fmt=st["marker"], ms=3.0, capsize=1.3, elinewidth=0.65, color=st["color"], ecolor=st["color"],
                         markeredgecolor="white", markeredgewidth=0.35, alpha=0.78, zorder=2)
        handles.append(h); labels.append(st["label"])
        c = curve(d, truth); curves[p] = c
        cov = (q16 <= truth) & (truth <= q84)
        slope = float(np.polyfit(truth, med, 1)[0])
        rows.append(dict(w=w, dataset_size=2**p, covered68=int(cov.sum()), coverage95=float(c[np.isclose(LEVELS, 0.95)][0]),
                         median_bias=float(np.median(med - truth)), median_width68=float(np.median(q84 - q16)), slope=slope))
    rax.set(xlim=lim, ylim=lim)
    rax.set_aspect("equal", adjustable="box")
    rax.set_xlabel(r"Requested $\Omega_m$", fontsize=10.5)
    rax.set_ylabel(r"Recovered $\Omega_m$", fontsize=10.5)
    style_axis(rax); rax.tick_params(axis="both", labelsize=9.0)

    cax.plot((0, 1), (0, 1), color="0.15", ls="--", lw=0.9, zorder=0)
    rc = curve(real, truth)
    for p, st in STYLES.items():
        cax.plot(LEVELS, curves[p], color=st["color"], lw=1.45, zorder=2)
        for lv, mk in FOCAL.items():
            cax.scatter([lv], curves[p][np.isclose(LEVELS, lv)], s=22, marker=mk, color=st["color"], edgecolor="0.15",
                        linewidth=0.45, zorder=3)
    cax.text(0.40, 0.90, "underconfident", color="0.38", fontsize=8.5, ha="center")
    cax.text(0.70, 0.24, "overconfident", color="0.38", fontsize=8.5, ha="center")
    cax.set_xlabel("Nominal coverage", fontsize=10.5); cax.set_ylabel("Empirical coverage", fontsize=10.5)
    cax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
    style_axis(cax); cax.tick_params(axis="both", labelsize=9.0)
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 0.98), ncol=3,
               handlelength=1.6, columnspacing=1.1, fontsize=8.5)
    rows.append(dict(w="real", dataset_size="real", covered68=int((np.quantile(real, 0.16, 1) <= truth).__and__(truth <= np.quantile(real, 0.84, 1)).sum()),
                     coverage95=float(rc[np.isclose(LEVELS, 0.95)][0]), median_bias=float(np.median(np.median(real, 1) - truth)),
                     median_width68=float(np.median(np.quantile(real, 0.84, 1) - np.quantile(real, 0.16, 1))),
                     slope=float(np.polyfit(truth, np.median(real, 1), 1)[0])))
    return fig, pd.DataFrame(rows), pd.DataFrame({"nominal": LEVELS, "real": rc, **{f"N{2**p}": curves[p] for p in STYLES}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, action="append", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / f"cfg_paper_recovery_figure_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    truth, d = load_real()
    for w in args.w:
        fig, summ, curves = figure_for(w, truth, d["real"])
        stem = f"conditional_recovery_cfg_{wlabel(w)}"
        fig.savefig(out / f"{stem}.pdf", format="pdf", dpi=300, facecolor="white")
        fig.savefig(out / f"{stem}.png", format="png", dpi=200, facecolor="white")
        plt.close(fig)
        summ.to_csv(out / f"{stem}_summary.csv", index=False)
        curves.to_csv(out / f"{stem}_coverage_curves.csv", index=False)
        print(summ.round(3).to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
