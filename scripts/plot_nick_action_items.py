#!/usr/bin/env python
"""Figures and tables for Nick Kern's action items (2026-10) plus the follow-up checks.

Reads only saved outputs (evaluate_run.py results, sample files, training images); computes nothing new except
(a) nearest-training-map pairs for the galleries (simdiff_eval.orbit_nn, D4 x periodic shift) and (b) the N=256 +0
per-k P(k) ratio, which no evaluate_run output stores (same definition as evaluate_run.py: per held-out cosmology mean
P_gen over draws / mean P_real over its real slices, geometric mean over the 32 cosmologies).
  1  CFG guidance sweep        results/cfg_results_by_w_20261012_002216/ (copied figures)
  2-3 augmentation wave 2       results/unet128_aug_wave2/evaluate_run_v1/
  4  aug fine-tune of trained models  results/nf_cond_omsigc_n256_aug_finetune/, results/nf_cond_omsigc_n1024_n4096_aug_finetune/
  +  DPM-50 timestep-grid test  results/dpm50_trailing_grid_test/
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

from plot_iaifi_statement_figure import INK, LEVELS, MUTED, TEST, load as load_real, style, summarize

ROOT = Path(__file__).resolve().parents[1]
W_REAL = 0.0931
REAL = "#9a9893"
ARMS = ["noaug", "fliponly", "d4only", "shiftonly", "d4shift"]
ARM_LABEL = {"noaug": "no augmentation", "fliponly": "flips only (2 views)", "d4only": "rotations + flips (D4, 8 views)",
             "shiftonly": "translations only", "d4shift": "D4 + translations"}
ARM_COLOR = {"noaug": "#52514e", "fliponly": "#1baf7a", "d4only": "#eb6834", "shiftonly": "#2a78d6", "d4shift": "#eda100"}
ARM_MARKER = {"noaug": "o", "fliponly": "s", "d4only": "D", "shiftonly": "^", "d4shift": "v"}
FT_COLOR = {0: "#52514e", 25: "#86b6ef", 50: "#2a78d6", 100: "#0d366b"}
N_COLOR = {256: "#2a78d6", 1024: "#eb6834", 4096: "#1baf7a"}

W2 = ROOT / "results/unet128_aug_wave2/evaluate_run_v1"
FT256 = ROOT / "results/nf_cond_omsigc_n256_aug_finetune/evaluate_run_v1"
FT1K = ROOT / "results/nf_cond_omsigc_n1024_n4096_aug_finetune"
GRID = ROOT / "results/dpm50_trailing_grid_test"
CFG_FIGS = ROOT / "results/cfg_results_by_w_20261012_002216"
BASE = "nf_cond_omsigc_hi_u128_{d}_n{n}_fresh200k"
DTAG = {256: "d2p08", 1024: "d2p10", 4096: "d2p12", 32768: "d2p15"}


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def orbit_pair(query, train):
    """Nearest training map of one generated map under D4 x periodic shift; returns aligned query, ref, cosine."""
    from simdiff_eval.orbit_nn import orbit_max_cosine

    r = orbit_max_cosine(query[None], train, device="cpu", query_batch=1, ref_batch=256)
    g, sy, sx, j = (int(r[k][0]) for k in ("group_element", "shift_y", "shift_x", "ref_index"))
    q = np.rot90(query, g % 4, axes=(0, 1))
    q = np.flip(q, -1) if g >= 4 else q
    return np.roll(q, (-sy, -sx), axis=(0, 1)), train[j], float(r["max_cosine"][0]), g, (sy, sx)


def show_pair(axes, gen, ref, cos, title):
    lo, hi = np.percentile(ref, [1, 99.5])
    for ax, im, lab in zip(axes, (gen, ref), ("generated", "nearest training")):
        ax.imshow(im, cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(lab, fontsize=6.5, pad=2)
    axes[0].set_ylabel(title, fontsize=6.8)
    axes[1].text(0.97, 0.04, f"cos {cos:.2f}", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=6.2,
                 color="white")


# ------------------------------------------------------------------ items 2-3: augmentation wave 2
def wave2(out):
    pr = pd.read_csv(W2 / "per_run.csv")
    cv = np.load(W2 / "curves.npz", allow_pickle=True)
    Ns = [64, 128, 256]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), gridspec_kw=dict(wspace=0.38, left=0.07, right=0.79,
                                                                         bottom=0.2, top=0.86))
    for key, ax, lab in (("orbit_copy_fraction", axes[0], "copy fraction\n(D4 × shift orbit)"),
                         ("G_d4", axes[1], "novelty score G\n(PCA, D4)"),
                         ("nearest_train_cosine_median", axes[2], "nearest-training cosine\n(plain, median)")):
        for arm in ARMS:
            d = pr[pr.arm == arm].sort_values("dataset_size")
            ax.plot(d.dataset_size, d[key], color=ARM_COLOR[arm], marker=ARM_MARKER[arm], ms=4, lw=1.4,
                    label=ARM_LABEL[arm])
        ax.set_xscale("log", base=2)
        ax.set(xticks=Ns, xticklabels=[str(n) for n in Ns], ylim=(-0.04, 1.04), xlabel="training maps N")
        ax.set_title(lab, fontsize=7.5)
    axes[0].axhline(0.5, color=MUTED, lw=0.6, ls=(0, (4, 3)))
    axes[1].axhline(0.5, color=MUTED, lw=0.6, ls=(0, (4, 3)))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(0.795, 0.52), fontsize=6.4,
               frameon=False)
    save(fig, out, "w2_memorization_vs_N")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex="row",
                             gridspec_kw=dict(wspace=0.12, hspace=0.45, left=0.08, right=0.79, bottom=0.1, top=0.93))
    for i, n in enumerate(Ns):
        ax = axes[0, i]
        kc = cv["kc"]
        ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        ax.plot(kc, cv[f"N{n}_reference__pk_ratio"], color=REAL, lw=2.2, alpha=0.6, label="real (train vs held-out)")
        for arm in ARMS:
            run = pr[(pr.arm == arm) & (pr.dataset_size == n)].run_name.iloc[0]
            ax.plot(kc, cv[f"{run}__pk_ratio"], color=ARM_COLOR[arm], lw=1.2, label=ARM_LABEL[arm])
        ax.set(xscale="log", ylim=(0.3, 1.7), title=f"P(k) ratio, N = {n}", xlabel="k (grid units)")
        if i:
            ax.set_yticklabels([])
        ax = axes[1, i]
        e = cv[f"N{n}_reference__pdf_edges"]
        c = 0.5 * (e[1:] + e[:-1])
        ax.fill_between(c, cv[f"N{n}_reference__pdf_val"], color=REAL, alpha=0.35, lw=0, label="held-out real")
        for arm in ARMS:
            run = pr[(pr.arm == arm) & (pr.dataset_size == n)].run_name.iloc[0]
            ax.plot(c, cv[f"{run}__pdf_gen"], color=ARM_COLOR[arm], lw=1.0)
        ax.set(yscale="log", title=f"one-point PDF, N = {n}", xlabel="normalized pixel value")
        if i:
            ax.set_yticklabels([])
    axes[0, 0].set_ylabel(r"$P_\mathrm{gen}/P_\mathrm{real}$")
    axes[1, 0].set_ylabel("density")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(0.795, 0.6),
               fontsize=6.4, frameon=False)
    save(fig, out, "w2_pk_onepoint")

    # gallery: the median-novelty sample of each arm at N = 64 and 128, aligned to its nearest training map
    from evaluate_run import UncondData, load_samples

    runs = {(r["arm"], r["dataset_size"]): r for r in json.loads((ROOT / "local/unet128_aug_wave2/eval_runs.json").read_text())}
    fig, axes = plt.subplots(len(ARMS), 4, figsize=(5.6, 7.2), gridspec_kw=dict(wspace=0.04, hspace=0.25, left=0.12,
                                                                                  right=0.99, bottom=0.01, top=0.93))
    gal = []
    for j, n in enumerate((64, 128)):
        data = None
        for i, arm in enumerate(ARMS):
            r = runs[(arm, n)]
            data = data if data is not None and data.config_path == ROOT / r["config"] else UncondData(ROOT / r["config"], 8, 64)
            gen = load_samples(ROOT / r["sample_path"].format(seed=123))
            nn = np.load(W2 / f"{r['run_name']}_per_sample.npz")["orbit_nn"]
            idx = int(np.argsort(nn)[len(nn) // 2])
            q, ref, cos, g, sh = orbit_pair(gen[idx], data.train)
            show_pair(axes[i, 2 * j:2 * j + 2], q, ref, cos, ARM_LABEL[arm] if j == 0 else "")
            gal.append(dict(arm=arm, N=n, sample_index=idx, orbit_cosine=cos, group_element=g, shift=list(sh)))
        fig.text(0.12 + 0.87 * (0.25 + 0.5 * j), 0.985, f"N = {n}", ha="center", va="top", fontsize=8.5)
    save(fig, out, "w2_gallery")
    return dict(per_run=pr[["arm", "dataset_size", "memorized", "orbit_copy_fraction", "orbit_copy_fraction_shift",
                            "orbit_copy_fraction_d4", "G_d4", "nearest_train_cosine_median", "pk_large", "pk_mid",
                            "pk_small", "one_point_l1"]].to_dict("records"), gallery=gal)


# ------------------------------------------------------------------ item 4: fine-tunes
def ft_runs():
    rows = []
    for n, steps, folder in ((256, (0, 25, 50, 100), FT256), (1024, (0, 25, 50), FT1K / "evaluate_run_v1"),
                             (4096, (0, 25, 50), FT1K / "evaluate_run_v1")):
        base = BASE.format(d=DTAG[n], n=n)
        for s in steps:
            if s == 0:
                if n == 256:
                    pred = ROOT / "results/nf_conditional_omsig_continuous_200k/calibration_vgg/bias_probe_per_sample_predictions.csv"
                    rows.append(dict(N=n, step=0, run=base, pred=pred, pk=None, per_run=None,
                                     samples=ROOT / f"results/nf_conditional_omsig_continuous_200k/samples/{base}_seed123_dpm50_heldout_k64.npz"))
                else:
                    f = FT1K / "evaluate_run_plus0"
                    rows.append(dict(N=n, step=0, run=base, pred=f / f"{base}_per_sample_predictions.csv",
                                     pk=f / f"{base}_pk.npz", per_run=f / "per_run.csv",
                                     samples=ROOT / f"results/nf_conditional_omsig_continuous_200k/samples/{base}_seed123_dpm50_heldout_k64.npz"))
            else:
                run = f"nf_cond_omsigc_hi_u128_{DTAG[n]}_n{n}_d4shift_ft{s:03d}k_from200k"
                rows.append(dict(N=n, step=s, run=run, pred=folder / f"{run}_per_sample_predictions.csv",
                                 pk=folder / f"{run}_pk.npz", per_run=folder / "per_run.csv",
                                 samples=folder.parent / f"samples/{run}_seed123_dpm50_heldout_k64.npz"))
    return rows


def draws_for(pred_csv, run, truth):
    d = pd.read_csv(pred_csv)
    d = d[(d.run_name == run) & (d.parameter == "Omega_m") & (d.guidance_label == "noguidance")]
    if "cfg_dropout" in d:
        d = d[d.cfg_dropout.fillna(0) == 0]
    d = d.set_index(["heldout_sim", "seed_index"]).sort_index()
    assert np.allclose(d.groupby(level=0).theta_in.first().loc[TEST], truth, atol=1e-5), run
    return d.theta_rec.loc[TEST].to_numpy().reshape(32, -1)


_REALPK = {}


def pk_curve_from_samples(sample_path):
    import simdiff_eval.run_metrics as M
    from simdiff_eval.probe_eval import load_heldout_real_slices
    from prepare_nf_conditional_u128_config import DATA_ROOT

    norm = dict(np.load(ROOT / "results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz", allow_pickle=True)["normalization"].item())
    b = M.pk_binning(128)
    if not _REALPK:
        real, _, sim, _ = load_heldout_real_slices(DATA_ROOT, TEST, 128, norm)
        pk = M.power_spectra(M.as_maps(real), b)
        _REALPK["mean"] = np.stack([pk[sim == s].mean(0) for s in TEST])
    with np.load(sample_path, allow_pickle=True) as z:
        gen = M.as_maps(z["samples"])
        k = int(z["samples_per_cosmology"])
    ratios = M.power_spectra(gen, b).reshape(32, k, -1).mean(1) / _REALPK["mean"]
    return np.asarray(b["kc"]), np.exp(np.log(ratios).mean(0))


def finetune(out, truth, real):
    rows = ft_runs()
    stats = []
    for r in rows:
        s = summarize(truth, draws_for(r["pred"], r["run"], truth))
        slope, icpt = np.polyfit(truth, s["med"], 1)
        rec = dict(N=r["N"], step=r["step"], run=r["run"], covered68=int(s["covered"].sum()),
                   bias_over_W=float(np.median(s["med"] - truth) / W_REAL), width68=float(np.median(s["q84"] - s["q16"])),
                   slope=float(slope), median_abs_error=float(np.median(np.abs(s["med"] - truth))))
        if r["per_run"] is not None:
            x = pd.read_csv(r["per_run"]).set_index("run_name").loc[r["run"]]
            rec.update({k: float(x[k]) for k in ("pk_large", "pk_mid", "pk_small", "nearest_train_cosine_median",
                                                 "pix_std_ratio_median")})
        else:
            j = json.loads((ROOT / "results/nf_conditioning_mechanism_n256/ledger_metrics_20261009.json").read_text())["arms"]["baseline_cross_attn"]
            rec.update({k: float(j[k]) for k in ("pk_large", "pk_mid", "pk_small", "nearest_train_cosine_median",
                                                 "pix_std_ratio_median")})
        if r["pk"] is not None:
            z = np.load(r["pk"])
            kc, curve = z["kc"], np.exp(np.log(z["ratios"]).mean(0))
        else:
            kc, curve = pk_curve_from_samples(r["samples"])
        r.update(s=s, kc=kc, curve=curve)
        stats.append(rec)
    st = pd.DataFrame(stats)

    # recovery grid
    cols = [0, 25, 50, 100]
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.6), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06, hspace=0.14, left=0.09, right=0.99, bottom=0.08, top=0.94))
    lim, ticks = (0.08, 0.52), [0.1, 0.3, 0.5]
    for i, n in enumerate((256, 1024, 4096)):
        for j, key in enumerate(["real"] + cols):
            ax = axes[i, j]
            ax.set(xlim=lim, ylim=lim, xticks=ticks, yticks=ticks, aspect="equal")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title("real maps" if key == "real" else ("before (+0)" if key == 0 else f"+{key}k updates"),
                             fontsize=7.5, pad=3)
            if j == 0:
                ax.set_ylabel(f"N = {n:,}\nrecovered", fontsize=6.8)
            if i == 2:
                ax.set_xlabel(r"true $\Omega_\mathrm{m}$", fontsize=6.8)
            match = [r for r in rows if r["N"] == n and r["step"] == key]
            if key != "real" and not match:
                ax.axis("off")
                continue
            s = real if key == "real" else match[0]["s"]
            c = REAL if key == "real" else FT_COLOR[key]
            ax.plot(lim, lim, color=MUTED, lw=0.6, ls=(0, (4, 3)))
            for m, face in ((s["covered"], c), (~s["covered"], "white")):
                ax.errorbar(truth[m], s["med"][m], yerr=[s["med"][m] - s["q16"][m], s["q84"][m] - s["med"][m]], fmt="o",
                            ms=2.2, mfc=face, mec=c, mew=0.6, ecolor=c, elinewidth=0.6, capsize=0)
            ax.text(0.04, 0.96, f"{int(s['covered'].sum())}/32\n{np.median(s['med'] - truth) / W_REAL:+.2f} W",
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=INK, linespacing=1.1)
    save(fig, out, "ft_recovery_grid")

    # P(k) curves
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.07, right=0.86, bottom=0.21, top=0.87))
    for ax, n in zip(axes, (256, 1024, 4096)):
        ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        for r in [r for r in rows if r["N"] == n]:
            ax.plot(r["kc"], r["curve"], color=FT_COLOR[r["step"]], lw=1.5 if r["step"] == 0 else 1.1,
                    label="before (+0)" if r["step"] == 0 else f"+{r['step']}k updates")
        ax.set(xscale="log", ylim=(0.3, 1.7), title=f"N = {n:,}", xlabel="k (grid units)")
    axes[0].set_ylabel(r"$P_\mathrm{gen}/P_\mathrm{real}$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.865, 0.55), fontsize=6.6, frameon=False)
    save(fig, out, "ft_pk")

    # metrics vs fine-tune updates
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.45, left=0.06, right=0.86,
                                                                         bottom=0.24, top=0.85))
    refs = {"bias_over_W": 0.0, "covered68": 27, "pk_large": 1.0, "nearest_train_cosine_median": 0.125}
    titles = {"bias_over_W": r"$\Omega_\mathrm{m}$ bias / W", "covered68": "covered 68% (of 32)",
              "pk_large": "P(k) large scales", "nearest_train_cosine_median": "nearest-training cosine"}
    for ax, key in zip(axes, refs):
        ax.axhline(refs[key], color=REAL, lw=2.0, alpha=0.6, label="real maps")
        for n in (256, 1024, 4096):
            d = st[st.N == n].sort_values("step")
            ax.plot(d.step, d[key], color=N_COLOR[n], marker="o", ms=3.5, lw=1.3, label=f"N = {n:,}")
        ax.set(title=titles[key], xlabel="fine-tune updates (k)", xticks=[0, 25, 50, 100])
        ax.title.set_fontsize(7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.865, 0.55), fontsize=6.6, frameon=False)
    save(fig, out, "ft_metrics_vs_updates")

    # gallery: N = 256 before / after, one held-out cosmology, nearest training map
    fig, axes = plt.subplots(2, 4, figsize=(5.6, 3.1), gridspec_kw=dict(wspace=0.04, hspace=0.3, left=0.1, right=0.99,
                                                                          bottom=0.01, top=0.86))
    from simdiff_eval.conditional_unet_diagnostics import normalize_raw_hi

    norm = dict(np.load(ROOT / "results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz", allow_pickle=True)["normalization"].item())
    raw = np.load(f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_omsig_continuous_200k/prepared_data/{BASE.format(d='d2p08', n=256)}_train_images.npy", mmap_mode="r")
    timgs = np.asarray(normalize_raw_hi(raw[:, 0], norm), dtype=np.float32).reshape(len(raw), 128, 128)
    train = timgs.shape
    gal = []
    for i, step in enumerate((0, 100)):
        r = [x for x in rows if x["N"] == 256 and x["step"] == step][0]
        with np.load(r["samples"], allow_pickle=True) as z:
            gen = np.asarray(z["samples"]).reshape(32, -1, 128, 128)
        for j, (cosmo, draw) in enumerate(((5, 0), (20, 0))):
            q, ref, cos, g, sh = orbit_pair(gen[cosmo, draw], timgs)
            show_pair(axes[i, 2 * j:2 * j + 2], q, ref, cos,
                      ("before (+0)" if step == 0 else "+100k updates") if j == 0 else "")
            gal.append(dict(step=step, heldout_sim=int(TEST[cosmo]), draw=draw, orbit_cosine=cos))
        for j, cosmo in enumerate((5, 20)):
            fig.text(0.1 + 0.89 * (0.25 + 0.5 * j), 0.99, f"held-out sim {TEST[cosmo]}", ha="center", va="top",
                     fontsize=7.5)
    save(fig, out, "ft_gallery_n256")
    return st, gal, train


# ------------------------------------------------------------------ grid test
def grid(out):
    m = json.loads((ROOT / "local/dpm50_trailing_grid_test/manifest.json").read_text())
    lab = {"a": "N = 32,768, no CFG", "b": "N = 32,768, CFG (w = 1)", "c": "N = 4,096, no CFG"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.07, right=0.86, bottom=0.21, top=0.87))
    rows = []
    for ax, r in zip(axes, m):
        ref_dir, ref_run = ROOT / r["reference_eval"], r["reference_eval_run_name"]
        lead = np.load(ref_dir / f"{ref_run}_pk.npz")
        trail = np.load(GRID / "evaluate_run_trailing" / f"{r['run_name']}_pk.npz")
        ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        rec = dict(arm=r["arm"], run=r["run_name"])
        for z, c, name in ((lead, "#52514e", "leading grid 450→9 (all runs)"), (trail, "#2a78d6", "trailing grid 499→9")):
            kc, curve = z["kc"], np.exp(np.log(z["ratios"]).mean(0))
            ax.plot(kc, curve, color=c, lw=1.3, label=name)
            rec[f"{'lead' if c == '#52514e' else 'trail'}_k3"] = float(curve[0])
            rec[f"{'lead' if c == '#52514e' else 'trail'}_highk6"] = float(np.exp(np.log(curve[-6:]).mean()))
        for d, tag in ((ref_dir, "lead"), (GRID / "evaluate_run_trailing", "trail")):
            x = pd.read_csv(d / "per_run.csv").set_index("run_name").loc[ref_run if tag == "lead" else r["run_name"]]
            rec[f"{tag}_bias_W"] = float(x.Omega_m_median_bias_over_W)
            rec[f"{tag}_cov68"] = int(x.Omega_m_covered68)
        ax.set(xscale="log", ylim=(0.5, 1.7), title=lab[r["arm"]], xlabel="k (grid units)")
        ax.title.set_fontsize(8)
        rows.append(rec)
    axes[0].set_ylabel(r"$P_\mathrm{gen}/P_\mathrm{real}$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(0.865, 0.55), fontsize=6.4,
               frameon=False)
    save(fig, out, "grid_pk")
    return rows


def main():
    out = ROOT / "results" / f"nick_action_items_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    style()
    for name in ("omega_m_recovery_grid", "pk_ratio_by_w", "coverage_by_w"):
        shutil.copy(CFG_FIGS / f"{name}.png", out / f"cfg_{name}.png")
    shutil.copy(CFG_FIGS / "summary_by_N_w.csv", out / "cfg_summary_by_N_w.csv")
    truth, d = load_real()
    real = summarize(truth, d["real"])
    w2 = wave2(out)
    st, ft_gal, _ = finetune(out, truth, real)
    st.to_csv(out / "finetune_summary.csv", index=False)
    g = grid(out)
    pd.DataFrame(g).to_csv(out / "grid_test_summary.csv", index=False)
    summary = dict(wave2=w2, finetune=st.to_dict("records"), finetune_gallery=ft_gal, grid_test=g,
                   real=dict(covered68=int(real["covered"].sum()), width68=float(np.median(real["q84"] - real["q16"]))))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(st.round(3).to_string(index=False))
    print(pd.DataFrame(g).round(3).to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
