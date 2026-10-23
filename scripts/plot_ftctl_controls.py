#!/usr/bin/env python
"""Figures for the fine-tune controls (ledger nf_cond_omsigc_ft_controls_300k), for item 4 of Nick's action-items page.

Arms, all resumed from the 200k conditional (Omega_m, sigma_8) UNet-128 checkpoints:
  noaug   no augmentation          N = 1024, 4096   +25k, +50k, +100k   results/nf_cond_omsigc_ft_controls_300k/
  idaug   identity augmentation    N = 1024         +25k, +50k, +100k   (same code path as d4shift, output = input)
  d4shift D4 + periodic shifts     N = 1024, 4096   +25k, +50k from results/nf_cond_omsigc_n1024_n4096_aug_finetune/,
                                                    +100k from the controls chain (resumed from the +50k checkpoint)
  +0      the 200k checkpoints     results/nf_cond_omsigc_n1024_n4096_aug_finetune/evaluate_run_plus0/

Reads only saved outputs (evaluate_run.py per_run.csv, *_pk.npz, *_per_sample_predictions.csv, sample files).
Computes, for plotting only: the one-point PDF of normalized pixel values (gen vs held-out real, saved to
one_point_pdf.npz) and nearest-training pairs for the galleries (simdiff_eval.orbit_nn, D4 x periodic shift).
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

from plot_iaifi_statement_figure import INK, MUTED, TEST, load as load_real, style, summarize
from plot_nick_action_items import FT_COLOR, REAL, W_REAL, draws_for, orbit_pair, save

ROOT = Path(__file__).resolve().parents[1]
CTL = ROOT / "results/nf_cond_omsigc_ft_controls_300k"
FT1K = ROOT / "results/nf_cond_omsigc_n1024_n4096_aug_finetune"
PREP = Path("/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_omsig_continuous_200k/prepared_data")
DTAG = {1024: "d2p10", 4096: "d2p12"}
STEPS = (0, 25, 50, 100)
ARMS = {1024: ("noaug", "idaug", "d4shift"), 4096: ("noaug", "d4shift")}
LABEL = {"noaug": "no augmentation", "idaug": "identity augmentation", "d4shift": "D4 + translations"}
COLOR = {"noaug": "#52514e", "idaug": "#d55181", "d4shift": "#eda100"}
LS = {"noaug": "-", "idaug": (0, (4, 2)), "d4shift": "-"}
MARK = {"noaug": "o", "idaug": "s", "d4shift": "v"}
NORM_NPZ = ROOT / "results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz"


def base(n):
    return f"nf_cond_omsigc_hi_u128_{DTAG[n]}_n{n}_fresh200k"


def run_info(n, arm, step):
    """Paths for one (N, arm, step). step 0 is the shared 200k checkpoint."""
    if step == 0:
        f, run = FT1K / "evaluate_run_plus0", base(n)
        samples = ROOT / f"results/nf_conditional_omsig_continuous_200k/samples/{run}_seed123_dpm50_heldout_k64.npz"
    else:
        run = f"nf_cond_omsigc_hi_u128_{DTAG[n]}_n{n}_{arm}_ft{step:03d}k_from200k"
        root = FT1K if (arm == "d4shift" and step < 100) else CTL
        f, samples = root / "evaluate_run_v1", root / f"samples/{run}_seed123_dpm50_heldout_k64.npz"
    return dict(N=n, arm=arm, step=step, run=run, folder=f, pk=f / f"{run}_pk.npz",
                pred=f / f"{run}_per_sample_predictions.csv", per_run=f / "per_run.csv", samples=samples)


def all_runs():
    return [run_info(n, a, s) for n in ARMS for a in ARMS[n] for s in STEPS]


def metrics_table(truth):
    keep = ["Omega_m_median_bias", "Omega_m_median_bias_over_W", "Omega_m_covered68", "Omega_m_coverage95",
            "Omega_m_median_width68", "Omega_m_recovery_slope", "pk_large", "pk_mid", "pk_small",
            "nearest_train_cosine_median", "draw_to_draw_cosine_median", "pix_std_ratio_median"]
    rows = []
    for r in all_runs():
        x = pd.read_csv(r["per_run"]).set_index("run_name").loc[r["run"]]
        rows.append(dict(N=r["N"], arm=r["arm"], step=r["step"], run=r["run"],
                         source=str(r["per_run"].relative_to(ROOT)), **{k: float(x[k]) for k in keep}))
    return pd.DataFrame(rows)


def fig_metrics(st, out):
    keys = {"Omega_m_median_bias_over_W": (r"$\Omega_\mathrm{m}$ median bias / W", 0.0),
            "Omega_m_covered68": ("covered 68% (of 32)", 27),
            "pk_large": ("P(k) ratio, large scales", 1.0),
            "nearest_train_cosine_median": ("nearest-training cosine", None)}
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True,
                             gridspec_kw=dict(wspace=0.42, hspace=0.32, left=0.08, right=0.83, bottom=0.12, top=0.9))
    for i, n in enumerate(ARMS):
        for j, (k, (title, ref)) in enumerate(keys.items()):
            ax = axes[i, j]
            if ref is not None:
                ax.axhline(ref, color=REAL, lw=2.0, alpha=0.6, label="real maps" if i == j == 0 else None)
            for arm in ARMS[n]:
                d = st[(st.N == n) & (st.arm == arm)].sort_values("step")
                ax.plot(d.step, d[k], color=COLOR[arm], ls=LS[arm], marker=MARK[arm], ms=3.4, lw=1.3, label=LABEL[arm])
            if i == 0:
                ax.set_title(title, fontsize=7.5)
            ax.set(xticks=list(STEPS), xticklabels=[f"{200 + s}k" for s in STEPS])
            if i == 1:
                ax.set_xlabel("total updates", fontsize=6.8)
            ax.tick_params(labelsize=6)
        axes[i, 0].set_ylabel(f"N = {n:,}", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.835, 0.55), fontsize=6.6, frameon=False)
    save(fig, out, "ctl_metrics_vs_updates")


def fig_recovery(truth, real, out):
    rows = [(n, a) for n in ARMS for a in ARMS[n]]
    fig, axes = plt.subplots(len(rows), 5, figsize=(7.2, 7.4), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06, hspace=0.12, left=0.13, right=0.99, bottom=0.06, top=0.96))
    lim, ticks = (0.08, 0.52), [0.1, 0.3, 0.5]
    for i, (n, arm) in enumerate(rows):
        for j, key in enumerate(("real",) + STEPS):
            ax = axes[i, j]
            ax.set(xlim=lim, ylim=lim, xticks=ticks, yticks=ticks, aspect="equal")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title("real maps" if key == "real" else ("start (+0)" if key == 0 else f"+{key}k updates"),
                             fontsize=7.5, pad=3)
            if j == 0:
                ax.set_ylabel(f"N = {n:,}\n{LABEL[arm]}\nrecovered", fontsize=6.4)
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
            ax.text(0.04, 0.96, f"{int(s['covered'].sum())}/32\n{np.median(s['med'] - truth) / W_REAL:+.2f} W",
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=INK, linespacing=1.1)
    save(fig, out, "ctl_recovery_grid")


def fig_pk(out):
    panels = [(n, a) for n in ARMS for a in ARMS[n]]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.2), sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.07, right=0.86, bottom=0.2, top=0.82))
    for ax, (n, arm) in zip(axes, panels):
        ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        z0 = np.load(run_info(n, arm, 100)["pk"])
        ax.plot(z0["kc"], np.exp(np.log(z0["split_half"]).mean(0)), color=REAL, lw=2.0, alpha=0.6,
                label="real split-half")
        for s in STEPS:
            z = np.load(run_info(n, arm, s)["pk"])
            ax.plot(z["kc"], np.exp(np.log(z["ratios"]).mean(0)), color=FT_COLOR[s], lw=1.5 if s == 0 else 1.1,
                    label="200k, trained" if s == 0 else f"{200 + s}k, fine-tuned")
        ax.set(xscale="log", ylim=(0.3, 1.7), xlabel="k (grid units)")
        ax.set_title(f"N = {n:,}\n{LABEL[arm]}", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel(r"$P_\mathrm{gen}/P_\mathrm{real}$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.865, 0.52), fontsize=6.4, frameon=False)
    save(fig, out, "ctl_pk")


def load_maps(path):
    import simdiff_eval.run_metrics as M

    with np.load(path, allow_pickle=True) as z:
        return np.asarray(M.as_maps(z["samples"]), dtype=np.float32)


def fig_pdf(real_maps, out):
    edges = np.linspace(-0.75, 1.15, 77)
    c = 0.5 * (edges[1:] + edges[:-1])
    hist = lambda x: np.histogram(np.clip(x.ravel(), edges[0], edges[-1]), edges, density=True)[0]
    hr = hist(real_maps)
    saved = dict(edges=edges, real_heldout=hr)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.6), sharex=True,
                             gridspec_kw=dict(hspace=0.08, wspace=0.18, height_ratios=(2.2, 1), left=0.08, right=0.75,
                                              bottom=0.12, top=0.91))
    for j, n in enumerate(ARMS):
        top, bot = axes[0, j], axes[1, j]
        top.fill_between(c, hr, color=REAL, alpha=0.4, lw=0, label="held-out real maps")
        bot.axhline(1, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        curves = [("trained model, 200k", 0, "noaug", INK, (0, (1, 1.5)))] + [(f"{LABEL[a]}, 300k", 100, a, COLOR[a], LS[a]) for a in ARMS[n]]
        for lab, s, a, col, ls in curves:
            h = hist(load_maps(run_info(n, a, s)["samples"]))
            saved[f"N{n}_{a if s else 'start'}_{s:03d}k"] = h
            top.plot(c, h, color=col, ls=ls, lw=1.1, label=lab)
            with np.errstate(divide="ignore", invalid="ignore"):
                bot.plot(c, np.where(hr > 1e-3, h / hr, np.nan), color=col, ls=ls, lw=1.0)
        top.set(yscale="log", ylim=(1e-3, 8))
        top.set_title(f"one-point PDF, N = {n:,}", fontsize=7.5)
        bot.set(ylim=(0, 2.5), xlabel="normalized pixel value")
        for ax in (top, bot):
            ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("gen / real", fontsize=6.8)
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    fig.legend(h + [x for x, y in zip(h2, l2) if y not in l], l + [y for y in l2 if y not in l], loc="center left",
               bbox_to_anchor=(0.755, 0.55), fontsize=6.3, frameon=False)
    save(fig, out, "ctl_one_point_pdf")
    np.savez(out / "one_point_pdf.npz", **saved)


def fig_gallery(n, real_maps, real_sim, out, cosmos=(5, 20)):
    from simdiff_eval.conditional_unet_diagnostics import normalize_raw_hi

    norm = dict(np.load(NORM_NPZ, allow_pickle=True)["normalization"].item())
    raw = np.load(PREP / f"{base(n)}_train_images.npy", mmap_mode="r")
    train = np.asarray(normalize_raw_hi(raw[:, 0], norm), dtype=np.float32).reshape(len(raw), 128, 128)
    rows = [("trained model\n200k", "noaug", 0)] + [(f"{LABEL[a]}\n300k total", a, 100) for a in ARMS[n]]
    nr = len(rows) + 1
    fig, axes = plt.subplots(nr, 6, figsize=(7.2, 1.25 * nr + 0.4),
                             gridspec_kw=dict(wspace=0.04, hspace=0.22, left=0.14, right=0.99, bottom=0.01, top=0.93))
    rec = []
    for jc, ci in enumerate(cosmos):
        sim = int(TEST[ci])
        rs = real_maps[real_sim == sim]
        lo, hi = np.percentile(rs, [1, 99.5])
        show = lambda ax, im: (ax.imshow(im, cmap="magma", vmin=lo, vmax=hi, interpolation="nearest"),
                               ax.set_xticks([]), ax.set_yticks([]))
        cols = axes[:, 3 * jc:3 * jc + 3]
        show(cols[0, 0], rs[0]), show(cols[0, 1], rs[len(rs) // 2])
        cols[0, 2].axis("off")
        cols[0, 0].set_title("slice 0", fontsize=6.5, pad=2), cols[0, 1].set_title(f"slice {len(rs) // 2}", fontsize=6.5, pad=2)
        for i, (lab, a, s) in enumerate(rows, start=1):
            gen = load_maps(run_info(n, a, s)["samples"]).reshape(32, -1, 128, 128)[ci]
            show(cols[i, 0], gen[0]), show(cols[i, 1], gen[1])
            q, ref, cos, g, sh = orbit_pair(gen[0], train)
            show(cols[i, 2], ref)
            cols[i, 2].text(0.97, 0.04, f"cos {cos:.2f}", transform=cols[i, 2].transAxes, ha="right", va="bottom",
                            fontsize=6.0, color="white")
            if i == 1:
                for k, t in enumerate(("draw 0", "draw 1", "nearest training\nto draw 0")):
                    cols[i, k].set_title(t, fontsize=6.3, pad=2)
            rec.append(dict(N=n, arm=a, step=s, heldout_sim=sim, draw=0, orbit_cosine=cos))
        fig.text(0.14 + 0.85 * (0.25 + 0.5 * jc), 0.985, f"held-out sim {sim}", ha="center", va="top", fontsize=7.5)
    axes[0, 0].set_ylabel("real\nheld-out", fontsize=6.6)
    for i, (lab, a, s) in enumerate(rows, start=1):
        axes[i, 0].set_ylabel(lab, fontsize=6.4)
    save(fig, out, f"ctl_gallery_n{n}")
    return rec


def main():
    from prepare_nf_conditional_u128_config import DATA_ROOT
    from simdiff_eval.probe_eval import load_heldout_real_slices
    import simdiff_eval.run_metrics as M

    out = ROOT / "results" / f"ftctl_controls_figures_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    style()
    truth, d = load_real()
    real = summarize(truth, d["real"])
    st = metrics_table(truth)
    st.to_csv(out / "controls_summary.csv", index=False)
    fig_metrics(st, out)
    fig_recovery(truth, real, out)
    fig_pk(out)
    norm = dict(np.load(NORM_NPZ, allow_pickle=True)["normalization"].item())
    rm, _, rsim, _ = load_heldout_real_slices(DATA_ROOT, TEST, 128, norm)
    rm = np.asarray(M.as_maps(rm), dtype=np.float32)
    fig_pdf(rm, out)
    gal = fig_gallery(1024, rm, np.asarray(rsim), out) + fig_gallery(4096, rm, np.asarray(rsim), out)
    pd.DataFrame(gal).to_csv(out / "gallery_nearest_train.csv", index=False)
    (out / "summary.json").write_text(json.dumps(dict(controls=st.to_dict("records"), gallery=gal,
                                                      real=dict(covered68=int(real["covered"].sum()))),
                                                 indent=2, default=float) + "\n")
    print(st[["N", "arm", "step", "Omega_m_median_bias_over_W", "Omega_m_covered68", "pk_large",
              "nearest_train_cosine_median"]].round(3).to_string(index=False))
    print(pd.DataFrame(gal).round(3).to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
