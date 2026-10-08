#!/usr/bin/env python
"""One-row figure for the IAIFI research statement: Omega_m recovery by a frozen regressor.

Continuous (Omega_m, sigma_8) conditional model, 32 held-out cosmologies (sims 900-931).
Panels: real held-out maps (128 slices each), generated N = 32768, generated N = 64 (64 draws each),
and the empirical coverage of central intervals for all three. Same probe for every panel.
Data: results/nf_conditional_omsig_continuous_200k/calibration_vgg/bias_probe_per_sample_predictions.csv and
results/nf_conditional_bias_probe/transform_controls/probe_transform_predictions.csv (identity rows), as in
results/camille_near_far_20261006_161945/make_plots.py. Writes a new timestamped folder; never overwrites.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP = "nf_conditional_omsig_continuous_200k"
TEST = np.arange(900, 932)
SIZES = (32768, 64)
PARAMS = "/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/params_LH_IllustrisTNG.txt"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#d9d8d4"
# Reference palette slots 1-2 (validated all-pairs) plus neutral ink for the real-map reference.
COLOR = {"real": "#52514e", 32768: "#2a78d6", 64: "#eb6834"}
TITLE = {"real": "Real held-out maps", 32768: "Generated, $N = 32{,}768$", 64: "Generated, $N = 64$"}
LEVELS = np.unique(np.r_[np.linspace(0, 1, 101), 0.68, 0.95])


def load():
    truth = np.loadtxt(PARAMS)[TEST, 0]
    pred = pd.read_csv(ROOT / "results" / SWEEP / "calibration_vgg" / "bias_probe_per_sample_predictions.csv")
    pred = pred[(pred.guidance_label == "noguidance") & (pred.cfg_dropout == 0) & (pred.parameter == "Omega_m")]
    draws = {n: pred[pred.dataset_size == n].set_index(["heldout_sim", "seed_index"]).theta_rec.loc[TEST]
             .to_numpy().reshape(32, 64) for n in SIZES}
    assert np.allclose(pred.groupby("heldout_sim").theta_in.first().loc[TEST], truth, atol=1e-5)
    probe = json.loads((ROOT / "results" / SWEEP / "calibration_vgg" / "bias_probe_eval_metadata.json").read_text())
    ctrl = ROOT / "results/nf_conditional_bias_probe/transform_controls"
    assert json.loads((ctrl / "manifest.json").read_text())["encoder"]["sha256"] == \
        hashlib.sha256(Path(probe["encoder_path"]).read_bytes()).hexdigest(), "real-map control used another probe"
    chunks = pd.read_csv(ctrl / "probe_transform_predictions.csv", chunksize=500_000,
                         usecols=["transform", "sim_index", "z_index", "parameter", "theta_pred"])
    real = pd.concat(c[(c["transform"] == "identity") & (c["parameter"] == "Omega_m")] for c in chunks)
    real = real[real.sim_index.isin(TEST)].sort_values(["sim_index", "z_index"])
    counts = real.groupby("sim_index").size()
    assert list(counts.index) == list(TEST) and counts.nunique() == 1, counts
    draws["real"] = real.theta_pred.to_numpy().reshape(32, -1)
    return truth, draws


def summarize(truth, d):
    med = np.median(d, axis=1)
    q16, q84 = np.quantile(d, [0.16, 0.84], axis=1)
    lo = np.quantile(d, (1 - LEVELS) / 2, axis=1)
    hi = np.quantile(d, (1 + LEVELS) / 2, axis=1)
    curve = ((lo <= truth) & (truth <= hi)).mean(axis=1)
    covered = (q16 <= truth) & (truth <= q84)
    return dict(med=med, q16=q16, q84=q84, curve=curve, covered=covered, n_draws=d.shape[1])


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "text.color": INK, "pdf.fonttype": 42, "savefig.dpi": 400})


def main():
    out = ROOT / "results" / f"iaifi_statement_figure_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    truth, draws = load()
    keys = ["real", 32768, 64]
    s = {k: summarize(truth, draws[k]) for k in keys}
    style()
    fig = plt.figure(figsize=(7.0, 2.25))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.16, 1.0], wspace=0.12, left=0.06, right=0.985, bottom=0.2,
                          top=0.86)
    lim = (0.08, 0.52)
    ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        r = s[k]
        order = np.argsort(truth)
        ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        miss = ~r["covered"]
        for mask, face in ((r["covered"], COLOR[k]), (miss, "white")):
            ax.errorbar(truth[mask], r["med"][mask], yerr=[r["med"][mask] - r["q16"][mask], r["q84"][mask] - r["med"][mask]],
                        fmt="o", ms=3.4, mfc=face, mec=COLOR[k], mew=0.9, ecolor=COLOR[k], elinewidth=0.9,
                        capsize=0, zorder=3)
        ax.set(xlim=lim, ylim=lim, xticks=ticks, yticks=ticks, aspect="equal")
        ax.set_title(TITLE[k], pad=5)
        ax.set_xlabel(r"True $\Omega_\mathrm{m}$")
        if i == 0:
            ax.set_ylabel(r"Recovered $\Omega_\mathrm{m}$")
        else:
            ax.set_yticklabels([])
        n_cov = int(r["covered"].sum())
        ax.text(0.05, 0.95, f"{n_cov}/32 inside\n68% interval", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color=INK, linespacing=1.2)
        if i == 0:
            kx, ky = 0.62, 0.17
            ax.plot(kx, ky + 0.07, "o", ms=3.4, color=COLOR[k], transform=ax.transAxes)
            ax.plot(kx, ky, "o", ms=3.4, mfc="white", mec=COLOR[k], mew=0.9, transform=ax.transAxes)
            ax.text(kx + 0.05, ky + 0.07, "covers truth", transform=ax.transAxes, va="center", fontsize=6.5, color=MUTED)
            ax.text(kx + 0.05, ky, "misses", transform=ax.transAxes, va="center", fontsize=6.5, color=MUTED)
        for j in order:
            rows.append(dict(panel=str(k), heldout_sim=int(TEST[j]), true_omega_m=truth[j], median=r["med"][j],
                             q16=r["q16"][j], q84=r["q84"][j], covered68=bool(r["covered"][j]), n_draws=r["n_draws"]))
    ax = fig.add_subplot(gs[0, 4])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    label = {"real": "Real maps", 32768: "$N = 32{,}768$", 64: "$N = 64$"}
    for k in keys:
        ax.plot(LEVELS, s[k]["curve"], color=COLOR[k], lw=1.6, solid_capstyle="round", zorder=3, label=label[k])
        y68 = s[k]["curve"][np.isclose(LEVELS, 0.68)][0]
        ax.plot(0.68, y68, "o", ms=4.2, color=COLOR[k], mec="white", mew=0.8, zorder=4)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=1.4, borderaxespad=0.2, labelspacing=0.3,
              handletextpad=0.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal",
           xticklabels=["0", "0.5", "1"], yticklabels=["0", "0.5", "1"])
    ax.set_xlabel("Nominal interval")
    ax.set_ylabel("Fraction containing truth")
    ax.set_title("Interval calibration", pad=5)
    fig.savefig(out / "omega_m_recovery_real_vs_generated.pdf")
    fig.savefig(out / "omega_m_recovery_real_vs_generated.png")
    pd.DataFrame(rows).to_csv(out / "plotted_points.csv", index=False)
    summary = {str(k): dict(covered68=int(s[k]["covered"].sum()), n=32, n_draws=s[k]["n_draws"],
                            coverage95=float(s[k]["curve"][np.isclose(LEVELS, 0.95)][0]),
                            median_bias=float(np.median(s[k]["med"] - truth)),
                            median_width68=float(np.median(s[k]["q84"] - s[k]["q16"]))) for k in keys}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
