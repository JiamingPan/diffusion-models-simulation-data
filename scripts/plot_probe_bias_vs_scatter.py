#!/usr/bin/env python
"""Why the Omega_m coverage of real maps sits above the diagonal: per-box median bias against slice scatter.

For each held-out simulation i (900-931) and its set of probe predictions y_ij (128 real z=0 slices, or 64 generated
draws): median m_i, bias b_i = m_i - theta_i, 68% bar [q16_i, q84_i], half-width s_i = (q84_i - q16_i) / 2.
The box is covered at 68% iff q16_i <= theta_i <= q84_i (about |b_i| < s_i when the spread is symmetric).
Same inputs as results/conditional_recovery_real_reference_*/ (paper Fig. 2 plus the real-map curve):
generated = results/nf_conditional_bias_fresh_full_sweep_200k/calibration_vgg/bias_probe_per_sample_predictions.csv,
real = scripts/plot_iaifi_statement_figure.load (transform_controls identity rows). Reads saved predictions only.
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

from plot_iaifi_statement_figure import INK, MUTED, TEST, load as load_real, style

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "results/nf_conditional_bias_fresh_full_sweep_200k/calibration_vgg/bias_probe_per_sample_predictions.csv"
SETS = ["real", 128, 1024, 16384]
LABEL = {"real": "real maps (128 slices)", 128: r"generated, $N_{2D}=2^{7}$", 1024: r"generated, $N_{2D}=2^{10}$",
         16384: r"generated, $N_{2D}=2^{14}$"}
COLOR = {"real": "#52514e", 128: "#c0392b", 1024: "#e67e22", 16384: "#0072b2"}  # matches paper Fig. 2
MARK = {"real": "o", 128: "o", 1024: "D", 16384: "s"}


def generated(truth):
    p = pd.read_csv(SAMPLES)
    p = p[(p.parameter == "Omega_m") & (p.guidance_label == "noguidance") & (p.cfg_dropout.fillna(0) == 0)]
    out = {}
    for n in SETS[1:]:
        d = p[p.dataset_size == n].set_index(["heldout_sim", "seed_index"]).sort_index()
        assert np.allclose(d.groupby(level=0).theta_in.first().loc[TEST], truth, atol=1e-5), n
        out[n] = d.theta_rec.loc[TEST].to_numpy().reshape(32, -1)
    return out


def per_box(truth, y, name):
    m = np.median(y, axis=1)
    q16, q84 = np.quantile(y, [0.16, 0.84], axis=1)
    return pd.DataFrame(dict(set=str(name), heldout_sim=TEST, truth=truth, n_predictions=y.shape[1], median=m,
                             q16=q16, q84=q84, bias=m - truth, half_width=(q84 - q16) / 2,
                             covered68=(q16 <= truth) & (truth <= q84)))


def real_coverage_figure(truth, y, out):
    """Real held-out maps only: the 1-1 plot with its bars, and the coverage curve with the exact construction."""
    levels = np.unique(np.r_[np.linspace(0, 1, 51), 0.68, 0.95])
    lo = np.quantile(y, (1 - levels) / 2, axis=1)  # (levels, boxes)
    hi = np.quantile(y, (1 + levels) / 2, axis=1)
    cov = ((lo <= truth) & (truth <= hi)).mean(axis=1)
    pd.DataFrame(dict(nominal=levels, empirical=cov, covered=(cov * len(truth)).round().astype(int))).to_csv(
        out / "real_coverage_curve.csv", index=False)
    med = np.median(y, axis=1)
    q16, q84 = np.quantile(y, [0.16, 0.84], axis=1)
    inside = (q16 <= truth) & (truth <= q84)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9), gridspec_kw=dict(wspace=0.35, left=0.1, right=0.98, bottom=0.17,
                                                                          top=0.88))
    ax = axes[0]
    ax.plot([0.08, 0.52], [0.08, 0.52], color=MUTED, lw=0.7, ls=(0, (4, 3)))
    for m, face in ((inside, "#52514e"), (~inside, "white")):
        ax.errorbar(truth[m], med[m], yerr=[med[m] - q16[m], q84[m] - med[m]], fmt="o", ms=3, mfc=face, mec="#52514e",
                    mew=0.7, ecolor="#52514e", elinewidth=0.7, capsize=0)
    ax.set(xlim=(0.08, 0.52), ylim=(0.08, 0.52), aspect="equal", xlabel=r"true $\Omega_\mathrm{m}$ of the box",
           ylabel=r"probe $\Omega_\mathrm{m}$ (128 slices)")
    ax.set_title("(a) real held-out maps, 1-to-1", fontsize=8)
    ax.text(0.1, 0.5, f"{int(inside.sum())}/32 bars contain the truth\nfilled = contains, open = misses", fontsize=6.3,
            va="top", color=INK)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.plot(levels, cov, color="#52514e", lw=1.4)
    for lv, mk in ((0.68, "o"), (0.95, "s")):
        c = float(cov[np.isclose(levels, lv)][0])
        ax.plot(lv, c, mk, ms=5, mfc="#52514e", mec="white", mew=0.6)
        xt = (0.74, 0.66) if lv == 0.68 else (0.62, 0.96)
        ax.annotate(f"{lv:.0%} range: {round(c * 32)}/32", (lv, c), xytext=xt, fontsize=6.3,
                    ha="left" if lv == 0.68 else "right", va="center", color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.text(0.62, 0.42, "diagonal: a calibrated\nposterior interval", fontsize=6.2, color=MUTED, ha="left")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), aspect="equal", xlabel="nominal level p (central range of the 128 readings)",
           ylabel="fraction of 32 boxes containing truth")
    ax.xaxis.label.set_fontsize(7)
    ax.set_title("(b) real held-out maps, coverage", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"real_map_coverage.{ext}", dpi=300)
    return cov, levels


def main():
    out = ROOT / "results" / f"probe_bias_vs_scatter_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    style()
    truth, d = load_real()
    ys = {"real": d["real"], **generated(truth)}
    boxes = pd.concat([per_box(truth, ys[k], k) for k in SETS], ignore_index=True)
    boxes.to_csv(out / "per_box.csv", index=False)
    summ = boxes.assign(abs_bias=boxes.bias.abs(), ratio=boxes.bias.abs() / boxes.half_width).groupby("set", sort=False).agg(
        covered68=("covered68", "sum"), median_bias=("bias", "median"), median_abs_bias=("abs_bias", "median"),
        median_half_width=("half_width", "median"), median_abs_bias_over_half_width=("ratio", "median")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.42, left=0.07, right=0.99,
                                                                          bottom=0.2, top=0.78, width_ratios=(1, 1.1, 1.1)))
    # (a) one real box: the 128 slice predictions, their median and 16-84% range, and the truth
    ax = axes[0]
    r = boxes[boxes.set == "real"].reset_index(drop=True)
    i = int(np.argsort(np.abs(r.truth - np.median(r.truth)))[0])
    y = ys["real"][i]
    ax.hist(y, bins=24, color="#c9c8c3", edgecolor="white", lw=0.4)
    ax.axvspan(r.q16[i], r.q84[i], color="#52514e", alpha=0.12, lw=0)
    ax.axvline(r["median"][i], color="#52514e", lw=1.2)
    ax.axvline(r.truth[i], color="#0072b2", lw=1.2, ls=(0, (3, 2)))
    ax.set(xlabel=r"probe $\Omega_\mathrm{m}$ per slice", ylabel="slices", yticks=[])
    ax.set_title(f"(a) one real box, sim {TEST[i]}", fontsize=8)
    top = ax.get_ylim()[1]
    ax.text(r["median"][i], top * 0.97, " median", color="#52514e", fontsize=6.5, va="top", ha="left")
    ax.text(r.truth[i], top * 0.83, "truth ", color="#0072b2", fontsize=6.5, va="top", ha="right")
    ax.annotate("", xy=(r.q16[i], top * 0.06), xytext=(r.q84[i], top * 0.06),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    ax.text(r.q84[i], top * 0.12, " 16–84%", color=INK, fontsize=6.2, ha="left", va="bottom")

    # (b) every box: |median bias| against half-width of the 68% bar
    ax = axes[1]
    lim = 0.2
    ax.fill_between([0, lim], [0, lim], color="#e9f1e7", lw=0, zorder=0)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.text(0.078, 0.004, "covered:\n|bias| < half-width", ha="right", va="bottom", fontsize=6.2, color="#3d6b35")
    for k in SETS:
        b = boxes[boxes.set == str(k)]
        ax.scatter(b.half_width, b.bias.abs(), s=10, marker=MARK[k], facecolor=COLOR[k], edgecolor="white", lw=0.3,
                   label=LABEL[k], zorder=3 if k == "real" else 2)
    ax.set(xlim=(0, 0.08), ylim=(0, lim), xlabel="half-width of 68% bar (scatter)", ylabel="|median − truth| (bias)")
    ax.set_title("(b) bias vs scatter, every box", fontsize=8)

    # (c) the ratio |bias| / half-width per box; covered (68%) count on the right
    ax = axes[2]
    for j, k in enumerate(SETS):
        b = boxes[boxes.set == str(k)]
        rr = (b.bias.abs() / b.half_width).to_numpy()
        jit = (np.random.default_rng(0).random(len(rr)) - 0.5) * 0.35
        ax.scatter(np.clip(rr, 0, 6), j + jit, s=8, marker=MARK[k], facecolor=COLOR[k], edgecolor="white", lw=0.3)
        ax.plot([np.median(rr)] * 2, [j - 0.3, j + 0.3], color=INK, lw=1.2)
        ax.text(6.25, j, f"{int(b.covered68.sum())}/32", va="center", fontsize=6.5, color=INK)
    ax.axvline(1, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.set(yticks=range(len(SETS)), yticklabels=["real", r"$2^{7}$", r"$2^{10}$", r"$2^{14}$"], xlim=(0, 6.9),
           xlabel="|bias| / half-width (clipped at 6)")
    ax.invert_yaxis()
    ax.set_title("(c) ratio per box; covered at 68%", fontsize=8)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.53, 1.0),
               handletextpad=0.2, columnspacing=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"probe_bias_vs_scatter.{ext}", dpi=300)
    cov, lv = real_coverage_figure(truth, ys["real"], out)
    print("real coverage at 0.02/0.68/0.95:", [float(cov[np.isclose(lv, x)][0]) for x in (0.02, 0.68, 0.95)])
    print(summ.round(4).to_string(index=False))
    print("example box", TEST[i])
    print("Wrote", out)


if __name__ == "__main__":
    main()
