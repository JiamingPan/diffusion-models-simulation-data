#!/usr/bin/env python
"""Insert section 6 (superposition tests) into the low-N diagnostics notebook.

Idempotent: if a cell already starts with the section-6 header, nothing changes.
Cells are inserted immediately before the '## Takeaways' markdown cell. Existing
cells and outputs are untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = "### 6. Superposition tests — is the fixed low-N image a mixture, or a degraded copy?"

MD_INTRO = HEADER + r"""

Section 5 showed that at low N every noise seed gives the same image, and that a positive
combination of the 16 parameter-nearest training maps explains about half of it. That is
compatible with **two different pictures**: a weighted mixture of several neighbours, or a
degraded copy of *one* neighbour plus structure the network adds. The tests below separate them.

- **6A (saved arrays):** R² when NNLS is offered only the 1, 2, 3 nearest maps versus all 16;
  a random 16-map basis that excludes the parameter-nearest maps; and a low-pass version
  (σ = 2, 4 px) to ask which scales are inherited. The single-map cos² is printed next to the
  k = 1 R² so it can be compared with the section-1 nearest-training cosine.
- **6B (needs new samples):** the N=64 model sampled along straight lines in parameter space
  between mutually nearest training cosmologies, and at its own training cosmologies.
  Produced by `scripts/slurm/sample_nf_conditional_theta_interp.sbatch`; PENDING until then.

Stated predictions (written before looking):
mixture ⇒ R² keeps rising from k=1 to k=3 and the two-endpoint R² along the line stays ≳ 0.8
with weights moving smoothly from (1,0) to (0,1); degraded copy ⇒ R² is flat after k=1, the
midpoint is poorly explained by the two endpoints, and cosine to the own training map at a
training θ is ≈ 1. Either way the random basis should fall to the real-map floor if θ selects
the map. No training, sampling, downloads or writes under results/ occur in these cells.
"""

CODE_6A = r'''from simdiff_eval.conditional_unet_superposition import superposition_controls_run
sp_rows = []
sp_sizes_requested = [64, 128, 256, 1024, 32768]
if not lb_scipy_ready or not ready:
    display(Markdown("**PENDING:** 6A needs SciPy and the saved conditional arrays (Great Lakes)."))
else:
    sp_grid = np.load(GRID, mmap_mode="r", allow_pickle=False)
    sp_scales = np.asarray(json.loads(SCALES.read_text())["std"], dtype=float)
    for row, sample_file, k in ready:
        N = int(row["dataset_size"])
        if N not in sp_sizes_requested:
            continue
        cfg = yaml.safe_load(path(row["config"]).read_text())
        sp_norm = cfg["data"]["norm_kwargs"]
        sp_pairs = pd.read_csv(path(row["selected_pairs_path"]))
        sp_train = np.load(path(row["prepared_image_path"]), mmap_mode="r", allow_pickle=False)
        sp_labels = np.load(path(row["train_raw_params_path"]), allow_pickle=False)
        sp_ids = np.atleast_1d(np.loadtxt(path(row["heldout_indices_path"]), dtype=np.int64))
        sp_theta = np.load(path(row["heldout_raw_params_path"]), allow_pickle=False)
        with np.load(sample_file, allow_pickle=False) as archive:
            if (int(archive["samples_per_cosmology"]) != k
                    or not np.array_equal(archive["heldout_indices"], sp_ids)
                    or not np.allclose(archive["theta_raw"], sp_theta, rtol=0, atol=1e-6)):
                raise ValueError("Saved sample labels disagree with manifest")
            sp_samples = np.asarray(archive["samples"], dtype=np.float32)
        sp_z = np.linspace(0, 127, k, dtype=int)
        sp_real = np.stack([normalize_raw_hi(sp_grid[int(sim), sp_z], sp_norm) for sim in sp_ids])
        result = superposition_controls_run(
            sp_samples, sp_train, sp_labels, sp_pairs.simulation_index.to_numpy(np.int64),
            sp_ids, sp_theta, sp_scales, sp_real, sp_norm, k=k, sigmas=(2.0, 4.0), seed=SEED,
            draws=(0,))
        sp_rows.extend(dict(r, N=N) for r in result)
        print(f"N={N:,}: {len(result)} control fits over {len(sp_ids)} cosmologies")
        del sp_samples, sp_real
sp_controls = pd.DataFrame(sp_rows)
if not sp_controls.empty:
    sp_summary = (sp_controls.groupby(["N", "test", "sigma", "k", "kind"])
                  .agg(median_r2=("r2", "median"), q16_r2=("r2", lambda v: np.quantile(v, .16)),
                       q84_r2=("r2", lambda v: np.quantile(v, .84)),
                       median_single_map_cos2=("single_map_cos2", "median"),
                       median_effective_components=("effective_components", "median"),
                       cosmologies=("heldout_sim", "nunique")).reset_index())
    display(sp_summary.round(3))
    sp_sizes = sorted(sp_controls.N.unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    colors = {"generated": "#b45b38", "held-out real": "#315f85"}
    # panel 1: top-k curve at each N (generated solid, real dashed)
    topk = sp_summary[sp_summary.test == "parameter-nearest top-k"]
    for N in sp_sizes:
        for kind, ls in [("generated", "-"), ("held-out real", "--")]:
            part = topk[(topk.N == N) & (topk.kind == kind)].sort_values("k")
            axes[0].plot(part.k, part.median_r2, ls, marker="o", color=colors[kind],
                         alpha=.35 + .65 * (N == sp_sizes[0]), label=f"{kind} N={N:,}" if N in (sp_sizes[0], sp_sizes[-1]) else None)
    axes[0].set(xscale="log", xlabel="Basis maps offered to NNLS (parameter-nearest order)",
                ylabel="Median R²", title="How many nearest maps are needed?", ylim=(-.02, 1.02))
    axes[0].set_xticks([1, 2, 3, 16], ["1", "2", "3", "all"])
    axes[0].legend(frameon=False, fontsize=8)
    # panel 2: nearest vs random basis, generated only
    near = topk[(topk.kind == "generated") & (topk.k == topk.groupby("N").k.transform("max"))].set_index("N")
    rand = sp_summary[(sp_summary.test.str.startswith("random")) & (sp_summary.kind == "generated")].set_index("N")
    realfloor = topk[(topk.kind == "held-out real") & (topk.k == topk.groupby("N").k.transform("max"))].set_index("N")
    x = np.log2(sp_sizes)
    axes[1].plot(x, near.loc[sp_sizes].median_r2, "o-", color="#b45b38", label="generated, parameter-nearest basis")
    axes[1].plot(x, rand.loc[sp_sizes].median_r2, "s--", color="#b45b38", alpha=.6, label="generated, random basis (nearest excluded)")
    axes[1].plot(x, realfloor.loc[sp_sizes].median_r2, "^-", color="#315f85", label="real map, parameter-nearest basis")
    axes[1].set_xticks(x, [f"{N:,}" for N in sp_sizes]); axes[1].set(xlabel="Training maps N", ylabel="Median R²",
                title="Does θ select the maps?", ylim=(-.02, 1.02)); axes[1].legend(frameon=False, fontsize=8)
    # panel 3: low-pass R2 vs sigma at each N
    low = sp_summary[sp_summary.test == "low-pass parameter-nearest"]
    full = topk[topk.k == topk.groupby("N").k.transform("max")].assign(sigma=0.0)
    both = pd.concat([full, low])
    for N in sp_sizes:
        for kind, ls in [("generated", "-"), ("held-out real", "--")]:
            part = both[(both.N == N) & (both.kind == kind)].sort_values("sigma")
            axes[2].plot(part.sigma, part.median_r2, ls, marker="o", color=colors[kind],
                         alpha=.35 + .65 * (N == sp_sizes[0]))
    axes[2].set(xlabel="Gaussian low-pass σ (pixels); 0 = no smoothing", ylabel="Median R² (all nearest maps)",
                title="Which scales are inherited?", ylim=(-.02, 1.02))
    for ax in axes: ax.grid(alpha=.15)
    fig.tight_layout(); plt.show()
    for N in sp_sizes:
        g = topk[(topk.N == N) & (topk.kind == "generated")].sort_values("k")
        r1, r3, rall = g.median_r2.iloc[0], g[g.k == 3].median_r2.iloc[0], g.median_r2.iloc[-1]
        rr = rand.loc[N].median_r2; fl = realfloor.loc[N].median_r2
        gain = rall - r1
        verdict = ("mixture-like (R² keeps rising after k=1)" if gain >= 0.15 else
                   "single-map-like (R² nearly flat after k=1)")
        select = "θ selects the maps" if rr <= fl + 0.1 else "random basis fits nearly as well: θ is NOT selecting"
        lb_prediction_rows.append({"test": f"6A: N={N}", "prediction": "mixture ⇒ R² rises ≥.15 from k=1 to all; random basis ≈ real floor",
            "observed": f"k=1 {r1:.2f}, k=3 {r3:.2f}, all {rall:.2f}; random {rr:.2f}; real floor {fl:.2f}",
            "check": f"{verdict}; {select}"})
else:
    display(Markdown("**PENDING:** 6A produced no rows."))
'''

MD_6B = r"""#### 6B — Straight lines in parameter space, and the training cosmologies themselves

For four pairs (A, B) of mutually nearest training cosmologies, the N=64 model is sampled at
θ(s) = (1−s)θ_A + sθ_B, s ∈ {0, ¼, ½, ¾, 1}, with four noise seeds each, and separately at all
64 training cosmologies with four seeds each. Endpoint maps x_A, x_B are the exact selected
training slices. For each s: cosine to x_A and x_B, NNLS weights and R² on the two-map basis
{x_A, x_B}, and the draw-to-draw cosine over seeds.

**Predictions.** Mixture: two-map R² ≳ 0.8 at every s and weights moving monotonically.
Lookup with hallucinated detail: R² high only at s = 0 and s = 1, low at s = ½. Both pictures
predict cosine ≈ 1 to the own training map at s = 0, 1 and at the training cosmologies (true
memorization); if that fails, the model is not memorizing exact maps at all and the whole
"copy" language should be dropped.
"""

CODE_6B = r'''from simdiff_eval.conditional_unet_superposition import theta_interpolation_analysis, training_theta_analysis
sp_interp = pd.DataFrame(); sp_train_theta = pd.DataFrame()
sp_dir = PROJECT / "results" / SWEEP / "samples_theta_interp"
sp_row = next((r for r, _, _ in ready if int(r["dataset_size"]) == 64), None) if ready else None
if sp_row is None or not sp_dir.is_dir():
    display(Markdown("**PENDING:** 6B needs `results/<sweep>/samples_theta_interp/` from "
                     "`scripts/slurm/sample_nf_conditional_theta_interp.sbatch` (GPU, minutes)."))
else:
    sp_cfg = yaml.safe_load(path(sp_row["config"]).read_text()); sp_norm = sp_cfg["data"]["norm_kwargs"]
    sp_train = np.load(path(sp_row["prepared_image_path"]), mmap_mode="r", allow_pickle=False)
    sp_stats = json.loads(SCALES.read_text())
    sp_mean, sp_std = np.asarray(sp_stats["mean"], float), np.asarray(sp_stats["std"], float)
    interp_file = sp_dir / f"{sp_row['run_name']}_theta_interp_seed{SEED}_dpm50.npz"
    train_file = sp_dir / f"{sp_row['run_name']}_train_theta_seed{SEED}_dpm50.npz"
    if interp_file.is_file():
        with np.load(interp_file, allow_pickle=False) as a:
            if str(a["run_name"]) != sp_row["run_name"] or str(a["guidance_label"]) != "noguidance":
                raise ValueError("interpolation samples name another run or use guidance")
            meta = json.loads(str(a["label_meta_json"])); samples = np.asarray(a["samples"], np.float32)
            if not np.allclose(a["labels_norm"], (a["labels_raw"] - sp_mean) / sp_std, atol=1e-5):
                raise ValueError("label normalization mismatch in saved samples")
        s_values = np.asarray(meta["s_values"], float)
        x_a = normalize_raw_hi(sp_train[[p["row_a"] for p in meta["pairs"]]], sp_norm)
        x_b = normalize_raw_hi(sp_train[[p["row_b"] for p in meta["pairs"]]], sp_norm)
        sp_interp = pd.DataFrame(theta_interpolation_analysis(samples, x_a, x_b, meta["index"], s_values))
        display(sp_interp.round(3))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for p, part in sp_interp.groupby("pair"):
            lab = f"sims {meta['pairs'][p]['sim_a']}–{meta['pairs'][p]['sim_b']}"
            axes[0].plot(part.s, part.weight_a, "o-", label=lab); axes[0].plot(part.s, part.weight_b, "s--", color=axes[0].lines[-1].get_color())
            axes[1].plot(part.s, part.two_map_r2, "o-", label=lab)
            axes[2].plot(part.s, part.cos_a, "o-", label=lab); axes[2].plot(part.s, part.cos_b, "s--", color=axes[2].lines[-1].get_color())
        axes[0].set(xlabel="s along θ_A → θ_B", ylabel="Normalized NNLS weight (circle A, square B)", title="Do the weights morph?")
        axes[1].set(xlabel="s", ylabel="R² on {x_A, x_B}", ylim=(-.02, 1.02), title="Is the line explained by its endpoints?")
        axes[2].set(xlabel="s", ylabel="Cosine to x_A (circle) / x_B (square)", ylim=(-.02, 1.02), title="Copies at the ends?")
        axes[1].axhline(.8, ls=":", color=".4"); axes[1].legend(frameon=False, fontsize=8)
        for ax in axes: ax.grid(alpha=.15)
        fig.tight_layout(); plt.show()
        # image strip for the first pair, seed 0
        p0 = meta["pairs"][0]; strip = [x_a[0]] + [samples[r["row"]] for r in meta["index"] if r["pair"] == 0 and r["seed"] == 0] + [x_b[0]]
        titles = [f"x_A (row {p0['row_a']})"] + [f"s={s:g}" for s in s_values] + [f"x_B (row {p0['row_b']})"]
        lim = np.quantile(np.concatenate([m.ravel() for m in strip]), [.01, .99])
        fig, axes = plt.subplots(1, len(strip), figsize=(2.6 * len(strip), 2.8))
        for ax, m, t in zip(axes, strip, titles):
            ax.imshow(m, vmin=lim[0], vmax=lim[1], cmap="viridis"); ax.set_title(t, fontsize=9); ax.axis("off")
        fig.suptitle(f"N=64 along θ_A → θ_B, sims {p0['sim_a']}–{p0['sim_b']}, seed 0"); fig.tight_layout(); plt.show()
        mid = sp_interp[np.isclose(sp_interp.s, .5)]; ends = sp_interp[np.isin(sp_interp.s, [0., 1.])]
        end_cos = np.median(np.where(ends.s == 0, ends.cos_a, ends.cos_b))
        lb_prediction_rows.append({"test": "6B: θ interpolation", "prediction": "mixture ⇒ two-map R² ≥.8 at s=½; both ⇒ endpoint cosine ≈1",
            "observed": f"median R² at s=½ {mid.two_map_r2.median():.2f}; endpoint cosine {end_cos:.2f}; draw-to-draw {sp_interp.draw_to_draw_cosine.median():.2f}",
            "check": ("mixture-like" if mid.two_map_r2.median() >= .8 else "not explained by endpoints at midpoint")
                     + ("; exact copies at ends" if end_cos >= .95 else "; NOT exact copies at ends")})
    else:
        display(Markdown(f"**PENDING:** {interp_file.name} not found."))
    if train_file.is_file():
        with np.load(train_file, allow_pickle=False) as a:
            meta_t = json.loads(str(a["label_meta_json"])); samples_t = np.asarray(a["samples"], np.float32)
        own = normalize_raw_hi(sp_train[np.asarray(meta_t["training_rows"])], sp_norm)
        sp_train_theta = pd.DataFrame(training_theta_analysis(samples_t, own, int(meta_t["seeds"])))
        display(sp_train_theta.describe().round(3))
        lb_prediction_rows.append({"test": "6B: training θ", "prediction": "memorization ⇒ cosine to own map ≈1 (held-out gives 0.67)",
            "observed": f"median cosine to own map {sp_train_theta.cos_to_own_map.median():.3f}; draw-to-draw {sp_train_theta.draw_to_draw_cosine.median():.3f}",
            "check": "copies at training θ" if sp_train_theta.cos_to_own_map.median() >= .95 else "NOT exact copies at training θ"})
    else:
        display(Markdown(f"**PENDING:** {train_file.name} not found."))
'''


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    nb = json.loads(args.notebook.read_text())
    if any("".join(c["source"]).startswith(HEADER) for c in nb["cells"]):
        print("section 6 already present; nothing changed")
        return
    anchor = next(i for i, c in enumerate(nb["cells"])
                  if c["cell_type"] == "markdown" and "".join(c["source"]).lstrip().startswith("## Takeaways"))
    new = [cell("markdown", MD_INTRO), cell("code", CODE_6A), cell("markdown", MD_6B), cell("code", CODE_6B)]
    nb["cells"][anchor:anchor] = new
    args.notebook.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"inserted {len(new)} cells before Takeaways in {args.notebook}")


if __name__ == "__main__":
    main()
