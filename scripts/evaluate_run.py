#!/usr/bin/env python
"""Ledger metric source: score saved samples of unconditional or conditional runs.

Modes
-----
unconditional  Memorization (PCA32 q95 novelty G with and without D4 search;
               D4 x periodic-shift pixel orbit copies; plain nearest-training
               cosine; draw-to-draw cosine), one-point statistics, P(k) ratio by
               band, and the probe-space parameter distribution of generated maps
               against the training and held-out real maps. Reference level for
               every distribution metric: training vs held-out real maps.
conditional    Probe recovery on held-out cosmologies (median bias, bias / W,
               68% and 95% coverage, recovery and coverage plots), P(k) bands per
               cosmology against the real held-out slices (real split-half as
               reference), one-point statistics, draw-to-draw and nearest-training
               cosine.
regression     Reproduces saved numbers before any new score is trusted: R1 PCA95 G
               for DiT-L16, R3 P(k) bands (continuous Omega_m-sigma_8, N=256),
               R4 probe recovery (continuous Omega_m-sigma_8, N=64). Writes
               regression_pass.json only if every check passes.

Every unconditional/conditional output records whether regression_pass.json
existed when it ran ("trusted"); untrusted outputs are still written.
Refuses to write into an existing --out-dir. Reads saved samples only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from simdiff_eval import run_metrics as rm  # noqa: E402

DEFAULT_PROBE = "results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz"
DEFAULT_RECEIPT = "results/evaluate_run_regression_v1/regression_pass.json"
DIT_PLANS = ["/scratch/huterer_root/huterer0/jiamingp/dit_l16_adalnzero_highn_screen_v1/plan.json",
             "/scratch/huterer_root/huterer0/jiamingp/dit_zero_remaining_300k_v1/plan.json"]


# ---------------------------------------------------------------- helpers
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def fresh_dir(path: Path) -> Path:
    if path.exists():
        raise SystemExit(f"Refusing to write into existing {path}; choose a new --out-dir.")
    path.mkdir(parents=True)
    return path


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(json_ready(obj), indent=2) + "\n")


def evenly(n_total: int, n_keep: int | None) -> np.ndarray:
    if n_keep is None or n_keep <= 0 or n_total <= n_keep:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, int(n_keep), dtype=np.int64)


def trust_status(receipt: Path) -> dict:
    ok = receipt.is_file()
    return {"trusted": ok, "regression_receipt": str(receipt),
            "note": "regression checks passed" if ok else
            "UNTRUSTED: regression_pass.json missing; numbers are not ledger results"}


# ---------------------------------------------------------------- unconditional data
class UncondData:
    """Training and held-out real maps of one unconditional config, in its fitted model space."""

    def __init__(self, config_path: Path, heldout_raw_per_source: int, max_heldout: int):
        import yaml
        from simdiff_eval.io import _full_normalization_stats, _normalize_reference_slices, _streaming_source_specs

        self.config_path = config_path
        self.data_cfg = yaml.safe_load(config_path.read_text())["data"]
        specs = _streaming_source_specs(self.data_cfg)
        self.center, self.xmax = _full_normalization_stats(specs, self.data_cfg)
        self.norm_kwargs = dict(self.data_cfg.get("norm_kwargs") or {})
        zthin = int(self.data_cfg.get("zthin", 1))
        train_raw, val_raw = [], []
        for spec in specs:
            arr, sel = spec["array"], np.asarray(spec["selected"])
            for s in range(0, len(sel), 64):
                train_raw.append(self._slices(arr, sel[s:s + 64], zthin))
            unused = np.setdiff1d(np.arange(len(arr)), sel)[:int(heldout_raw_per_source)]
            if len(unused):
                val_raw.append(self._slices(arr, np.sort(unused), zthin))
        self.train_raw = np.concatenate(train_raw)
        val_raw = np.concatenate(val_raw)
        self.val_raw = val_raw[evenly(len(val_raw), max_heldout)]
        norm = lambda raw: _normalize_reference_slices(raw, self.data_cfg, self.center, self.xmax)[:, 0]
        self.train = np.concatenate([norm(self.train_raw[s:s + 4096]) for s in range(0, len(self.train_raw), 4096)])
        self.val = norm(self.val_raw)

    @staticmethod
    def _slices(arr, idx, zthin):
        raw = np.asarray(arr[np.asarray(idx)], dtype=np.float32)
        if raw.ndim == 4:
            return raw[:, ::zthin].reshape(-1, *raw.shape[-2:])
        return raw

    def to_probe_space(self, model_maps: np.ndarray, probe_norm: dict) -> np.ndarray:
        from train_nf_conditional_bias_encoder import tanh_normalize_logged

        log = rm.tanh_inverse_to_log(rm.as_maps(model_maps), self.center, self.xmax, self.norm_kwargs)
        return tanh_normalize_logged(log.astype(np.float32), probe_norm)


def load_pca_fit_maps(config: Path, max_maps: int) -> tuple[np.ndarray, int]:
    """Evenly spaced maps of one config's training set (PCA95 notebook cell 5)."""
    from simdiff_eval.io import configured_training_reference_info, iter_real_reference_batches_from_config

    total = int(configured_training_reference_info(config)["configured_slices"])
    idx = evenly(total, max_maps)
    parts, offset = [], 0
    for batch in iter_real_reference_batches_from_config(config):
        sel = idx[(idx >= offset) & (idx < offset + len(batch))]
        if len(sel):
            parts.append(np.asarray(batch[sel - offset], dtype=np.float32))
        offset += len(batch)
    if offset != total:
        raise ValueError(f"{config}: streamed {offset} maps, expected {total}")
    return rm.as_maps(np.concatenate(parts)), total


def load_samples(path: Path, max_n: int | None = None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        x = rm.as_maps(z["samples"], str(path))
    return x[:max_n] if max_n else x


def unconditional_rows(args) -> list[dict]:
    if args.preset == "unet128_aug_sweep":
        aug = json.loads((ROOT / "local/unet128_aug_sweep/runs.json").read_text())
        fig2 = json.loads((ROOT / "local/nf_generalize_fig2/manifest.json").read_text())
        rows = [dict(run_name=r["run_name"], arm=r["arm_tag"], dataset_size=int(r["dataset_size"]),
                     config=r["config"], sample_path=r["sample_path"].format(seed=args.seed)) for r in aug]
        rows += [dict(run_name=r["run_name"], arm="noaug", dataset_size=int(r["dataset_size"]), config=r["config"],
                      sample_path=r["sample_path"].format(seed=args.seed, sample_label=args.sample_label))
                 for r in fig2 if r["arch"] == "u128"]
    elif args.runs_json:
        rows = []
        for r in json.loads(Path(args.runs_json).read_text()):
            rows.append(dict(run_name=r["run_name"], arm=r.get("arm", r.get("arm_tag", "run")),
                             dataset_size=int(r["dataset_size"]), config=r["config"],
                             sample_path=r["sample_path"].format(seed=args.seed, sample_label=args.sample_label)))
    else:
        raise SystemExit("unconditional mode needs --preset or --runs-json")
    if args.run_name:
        rows = [r for r in rows if r["run_name"] in set(args.run_name)]
    if args.dataset_size:
        rows = [r for r in rows if r["dataset_size"] in set(args.dataset_size)]
    if not rows:
        raise SystemExit("No runs selected.")
    return sorted(rows, key=lambda r: (r["dataset_size"], r["arm"]))


def load_probe(path: Path, device: str):
    from evaluate_nf_conditional_bias_probe import load_vgg_encoder

    enc = load_vgg_encoder(ROOT, path, device)
    with np.load(path, allow_pickle=True) as d:
        norm = dict(d["normalization"].item())
        names = [str(x) for x in d["param_names"]]
    return enc, norm, names


PROBE_BATCH = 32


def probe_predict(enc, maps: np.ndarray) -> np.ndarray:
    return enc.norm_to_raw(enc.predict_norm(rm.as_maps(maps)[:, None], batch_size=PROBE_BATCH))


def run_unconditional(args) -> None:
    import pandas as pd

    out = fresh_dir(resolve(args.out_dir))
    rows = unconditional_rows(args)
    trust = trust_status(resolve(args.regression_receipt))
    print(json.dumps(trust), flush=True)
    t0 = time.time()
    fit_config = resolve(args.pca_fit_config)
    fit_maps, fit_total = load_pca_fit_maps(fit_config, args.pca_fit_max_real)
    space = rm.PCASpace(fit_maps, args.pca_components)
    print(f"PCA fit on {len(fit_maps)} of {fit_total} maps of {fit_config}; "
          f"{space.pca.n_components_} components, explained {space.explained_variance:.4f}", flush=True)
    del fit_maps
    probe = None if args.no_probe else load_probe(resolve(args.probe), args.device)
    binning = rm.pk_binning(128)
    summary, curves, provenance_runs = [], {}, []
    by_config: dict[str, list[dict]] = {}
    for r in rows:
        by_config.setdefault(r["config"], []).append(r)
    for config, group in by_config.items():
        n = group[0]["dataset_size"]
        print(f"[{time.time() - t0:7.0f}s] N={n}: loading {config}", flush=True)
        data = UncondData(resolve(config), args.heldout_raw_per_source, args.max_heldout)
        if len(data.train) != n:
            raise ValueError(f"{config}: {len(data.train)} training maps, manifest says {n}")
        ref = reference_block(data, binning, probe, args)
        curves[f"N{n}_reference"] = ref.pop("_curves")
        for r in group:
            path = resolve(r["sample_path"])
            base = dict(r, n_train=n, n_heldout=len(data.val),
                        **{f"ref_{k}": v for k, v in ref.items() if not k.startswith("_")})
            if not path.is_file():
                summary.append(dict(base, status="missing samples"))
                print(f"  {r['run_name']}: missing {path}", flush=True)
                continue
            gen = load_samples(path, args.max_generated)
            res, per_sample, cur = score_unconditional(data, gen, space, binning, probe, ref, args)
            summary.append(dict(base, status="ok", n_generated=len(gen), **res))
            curves[r["run_name"]] = cur
            np.savez_compressed(out / f"{r['run_name']}_per_sample.npz", **per_sample)
            provenance_runs.append(dict(r, sample_sha256=sha256(path), config_sha256=sha256(resolve(config))))
            print(f"  [{time.time() - t0:7.0f}s] {r['run_name']}: G={res['G_none']:.3f} G_d4={res['G_d4']:.3f} "
                  f"orbit_copy={res.get('orbit_copy_fraction', float('nan')):.3f} memorized={res['memorized']}",
                  flush=True)
        del data
    table = pd.DataFrame(summary)
    table.to_csv(out / "per_run.csv", index=False)
    np.savez_compressed(out / "curves.npz", kc=binning["kc"],
                        **{f"{k}__{kk}": v for k, c in curves.items() for kk, v in c.items()})
    plot_unconditional(table, curves, binning, out)
    metrics = {**trust, "mode": "unconditional", "per_run_csv": str(out / "per_run.csv"),
               "runs": summary}
    if args.preset == "unet128_aug_sweep":
        metrics["transition"] = transition_table(table)
    write_json(out / "metrics.json", metrics)
    write_json(out / "provenance.json", {
        "git": git_rev(), "argv": sys.argv, "pca_fit_config": str(fit_config),
        "pca_fit_config_sha256": sha256(fit_config), "pca_fit_maps": int(space.n_fit), "pca_fit_total": fit_total,
        "pca_components": int(space.pca.n_components_), "pca_explained_variance": space.explained_variance,
        "copy_quantile": args.copy_quantile, "orbit_max_n": args.orbit_max_n,
        "orbit_threshold_queries": args.orbit_threshold_queries, "probe": None if args.no_probe else str(args.probe),
        "probe_note": "probe trained on LH z=0 maps; on unconditional (LH+CV, z=0,1,2) maps it is a fixed "
                      "summary statistic compared across sets, not cosmology inference",
        "runs": provenance_runs, "elapsed_s": time.time() - t0, **trust})
    print("Wrote", out, flush=True)


def reference_block(data: UncondData, binning: dict, probe, args) -> dict:
    """Training vs held-out real: the reference level for P(k), one-point, probe and orbit metrics."""
    pk_train, pk_val = rm.power_spectra(data.train, binning), rm.power_spectra(data.val, binning)
    ratio = pk_train.mean(0) / pk_val.mean(0)
    edges = rm.one_point_edges(data.train, data.val)
    res = {**{k: v for k, v in rm.pk_band_values(ratio, binning).items() if not k.endswith("pct")},
           "one_point_l1": rm.one_point_l1(data.train, data.val, edges)["l1"],
           **{f"heldout_{k}": v for k, v in rm.one_point_moments(data.val).items()},
           **{f"train_{k}": v for k, v in rm.one_point_moments(data.train).items()},
           "draw_to_draw_cosine_heldout": rm.draw_to_draw_cosine(data.val)["mean"],
           "_edges": edges, "_pk_val_mean": pk_val.mean(0)}
    curves = {"pk_ratio": ratio, "pdf_val": np.histogram(data.val.ravel(), edges, density=True)[0],
              "pdf_train": np.histogram(data.train.ravel(), edges, density=True)[0], "pdf_edges": edges}
    if probe is not None:
        from train_nf_conditional_bias_encoder import preprocess_real_slices

        enc, norm, names = probe
        tr_idx = evenly(len(data.train), args.probe_max_train)
        train_probe = probe_predict(enc, data.to_probe_space(data.train[tr_idx], norm))
        direct = probe_predict(enc, preprocess_real_slices(data.train_raw[tr_idx[:64]], norm))
        val_probe = probe_predict(enc, data.to_probe_space(data.val, norm))
        res["probe_roundtrip_max_abs_dev"] = float(np.max(np.abs(train_probe[:64] - direct)))
        for p, name in enumerate(names[:2]):
            res[f"probe_{name}_W1_heldout_vs_train"] = rm.wasserstein1(val_probe[:, p], train_probe[:, p])
            res[f"probe_{name}_median_train"] = float(np.median(train_probe[:, p]))
            res[f"probe_{name}_median_heldout"] = float(np.median(val_probe[:, p]))
        curves["probe_train"], curves["probe_val"] = train_probe[:, :2], val_probe[:, :2]
        res["_probe_train"] = train_probe
    res["_curves"] = curves
    return res


def score_unconditional(data, gen, space, binning, probe, ref, args):
    res, per_sample, cur = {}, {}, {}
    for orbit in ("none", "d4"):
        nov = rm.pca_novelty(space, data.train, gen, args.copy_quantile, orbit=orbit)
        for k in ("G", "copy_fraction", "threshold", "generated_nn_median", "real_nn_median"):
            res[f"{k}_{orbit}"] = nov[k]
        per_sample[f"pca_gen_nn_{orbit}"], per_sample[f"pca_real_nn_{orbit}"] = nov["generated_nn"], nov["real_nn"]
    plain = rm.plain_nearest_cosine(data.train, gen)
    res["nearest_train_cosine_median"] = float(np.median(plain))
    per_sample["nearest_train_cosine"] = plain
    d2d = rm.draw_to_draw_cosine(gen)
    res["draw_to_draw_cosine_mean"], res["draw_to_draw_near_duplicate_fraction"] = d2d["mean"], d2d["near_duplicate_fraction"]
    if len(data.train) <= args.orbit_max_n:
        orb = rm.orbit_copy_stats(data.train, gen, data.val, args.copy_quantile, args.orbit_threshold_queries,
                                  device=None if args.device == "auto" else args.device)
        for k, v in orb.items():
            (per_sample if isinstance(v, np.ndarray) else res)[k] = v
    else:
        res["orbit_copy_fraction"] = float("nan")
        res["orbit_note"] = f"orbit search skipped: N > --orbit-max-n {args.orbit_max_n}"
    res["memorized"] = rm.memorized(res["G_d4"], res.get("orbit_copy_fraction"))
    pk_gen = rm.power_spectra(gen, binning).mean(0)
    ratio = pk_gen / ref["_pk_val_mean"]
    res.update({k: v for k, v in rm.pk_band_values(ratio, binning).items() if not k.endswith("pct")})
    op = rm.one_point_l1(gen, data.val, ref["_edges"])
    res["one_point_l1"], res["one_point_outside_bins"] = op["l1"], op["outside_a"]
    res.update({f"gen_{k}": v for k, v in rm.one_point_moments(gen).items()})
    cur["pk_ratio"] = ratio
    cur["pdf_gen"] = np.histogram(gen.ravel(), ref["_edges"], density=True)[0]
    if probe is not None:
        enc, norm, names = probe
        gp = probe_predict(enc, data.to_probe_space(gen, norm))
        per_sample["probe_pred"] = gp
        cur["probe_gen"] = gp[:, :2]
        for p, name in enumerate(names[:2]):
            res[f"probe_{name}_W1_gen_vs_train"] = rm.wasserstein1(gp[:, p], ref["_probe_train"][:, p])
            res[f"probe_{name}_median_gen"] = float(np.median(gp[:, p]))
    return res, per_sample, cur


def transition_table(table) -> list[dict]:
    """Smallest grid N not memorized, per arm (ledger unet128_aug_sweep_eval)."""
    out = []
    ok = table[table.status == "ok"]
    for arm, g in ok.groupby("arm"):
        g = g.sort_values("dataset_size")
        not_mem = g[~g.memorized.astype(bool)]
        out.append({"arm": arm, "sizes_scored": g.dataset_size.tolist(),
                    "memorized_by_N": dict(zip(g.dataset_size.astype(int), g.memorized.astype(bool))),
                    "N_T": int(not_mem.dataset_size.min()) if len(not_mem) else None})
    return out


def plot_unconditional(table, curves, binning, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = table[table.status == "ok"]
    if ok.empty:
        return
    arms = list(dict.fromkeys(ok.arm))
    colors = dict(zip(arms, ["#333333", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]))
    panels = [("G_none", "PCA32 G (no orbit)"), ("G_d4", "PCA32 G (D4 search)"),
              ("orbit_copy_fraction", "Pixel orbit copy fraction"), ("draw_to_draw_cosine_mean", "Draw-to-draw cosine"),
              ("pk_large", "P(k) ratio, large"), ("pk_mid", "P(k) ratio, mid"), ("pk_small", "P(k) ratio, small"),
              ("one_point_l1", "One-point PDF L1")]
    if "probe_Omega_m_W1_gen_vs_train" in ok:
        panels.append(("probe_Omega_m_W1_gen_vs_train", "Probe Ωm W1 to training"))
    ref_cols = {"pk_large": "ref_pk_large", "pk_mid": "ref_pk_mid", "pk_small": "ref_pk_small",
                "one_point_l1": "ref_one_point_l1", "draw_to_draw_cosine_mean": "ref_draw_to_draw_cosine_heldout",
                "orbit_copy_fraction": "heldout_orbit_copy_fraction",
                "probe_Omega_m_W1_gen_vs_train": "ref_probe_Omega_m_W1_heldout_vs_train"}
    ncol = 3
    fig, axes = plt.subplots(int(np.ceil(len(panels) / ncol)), ncol, figsize=(4.6 * ncol, 3.4 * np.ceil(len(panels) / ncol)),
                             constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), panels):
        for arm in arms:
            g = ok[ok.arm == arm].sort_values("dataset_size")
            if col in g:
                ax.plot(g.dataset_size, g[col], "o-", color=colors[arm], label=arm, lw=1.6, ms=4)
        rc = ref_cols.get(col)
        if rc and rc in ok:
            g = ok.drop_duplicates("dataset_size").sort_values("dataset_size")
            ax.plot(g.dataset_size, g[rc], "kx:", lw=1.2, label="real held-out reference")
        if col.startswith("G_") or col == "orbit_copy_fraction":
            ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
            ax.set_ylim(-0.03, 1.03)
        if col.startswith("pk_"):
            ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training maps N")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("evaluate_run.py unconditional summary (reference: training vs held-out real maps)", fontsize=11)
    fig.savefig(out / "summary_vs_N.png", dpi=160)
    plt.close(fig)

    kc = binning["kc"]
    for n, g in ok.groupby("dataset_size"):
        ref = curves.get(f"N{n}_reference", {})
        fig, axes = plt.subplots(1, 3 if "probe_val" in ref else 2, figsize=(15, 4), constrained_layout=True)
        for _, row in g.iterrows():
            c = curves.get(row.run_name, {})
            if "pk_ratio" in c:
                axes[0].plot(kc, c["pk_ratio"], color=colors[row.arm], label=row.arm)
            if "pdf_gen" in c:
                mid = 0.5 * (ref["pdf_edges"][1:] + ref["pdf_edges"][:-1])
                axes[1].plot(mid, c["pdf_gen"], color=colors[row.arm], label=row.arm)
            if "probe_gen" in c and len(axes) > 2:
                axes[2].hist(c["probe_gen"][:, 0], bins=30, histtype="step", color=colors[row.arm], density=True,
                             label=row.arm)
        if "pk_ratio" in ref:
            axes[0].plot(kc, ref["pk_ratio"], "k:", label="train / held-out")
            mid = 0.5 * (ref["pdf_edges"][1:] + ref["pdf_edges"][:-1])
            axes[1].plot(mid, ref["pdf_val"], "k-", lw=2, alpha=0.4, label="held-out real")
            axes[1].plot(mid, ref["pdf_train"], "k:", label="training")
        if len(axes) > 2:
            axes[2].hist(ref["probe_val"][:, 0], bins=30, histtype="stepfilled", color="0.8", density=True,
                         label="held-out real")
            axes[2].hist(ref["probe_train"][:, 0], bins=30, histtype="step", color="k", ls=":", density=True,
                         label="training")
            axes[2].set(xlabel="probe Ωm (fixed summary statistic)", title="Probe-space Ωm distribution")
        axes[0].axhline(1, color="0.6", lw=0.8, ls="--")
        axes[0].set(xscale="log", xlabel="k [grid units]", ylabel="P_gen / P_held-out", title=f"N = {n}: P(k) ratio")
        axes[1].set(yscale="log", xlabel="model-space pixel value", ylabel="PDF", title="One-point PDF")
        for ax in axes:
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
        fig.savefig(out / f"distributions_N{n}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- conditional
def conditional_scores(row: dict, args, enc, probe_norm: dict, names: list[str], out: Path | None) -> dict:
    """Probe recovery + P(k) + one-point + cosines for one conditional run (cosmology-major samples)."""
    import yaml
    from evaluate_nf_conditional_bias_probe import evaluate_run as probe_run, output_path_for
    from prepare_nf_conditional_u128_config import DATA_ROOT
    from simdiff_eval.conditional_unet_diagnostics import balanced_draws, recovery_interval_coverage
    from simdiff_eval.io import _normalize_reference_slices
    from simdiff_eval.probe_eval import load_heldout_real_slices

    cfg = yaml.safe_load(resolve(row["config"]).read_text())["data"]
    nk = cfg.get("norm_kwargs") or {}
    for key in ("center", "xmax"):
        if not np.isclose(float(nk[key]), float(probe_norm[key]), rtol=0, atol=1e-9):
            raise ValueError(f"{row['run_name']}: model {key} {nk[key]} != probe {probe_norm[key]}; "
                             "P(k)/one-point would compare different spaces")
    path = output_path_for(ROOT, row, args.seed, args.k, None)
    with np.load(path, allow_pickle=True) as z:
        samples = rm.as_maps(z["samples"])
        heldout = np.asarray(z["heldout_indices"], dtype=np.int64)
        k = int(z["samples_per_cosmology"])
    sample_df, points = probe_run(project_dir=ROOT, row=row, encoder=enc, seed=args.seed, k=k,
                                  embedding_batch_size=PROBE_BATCH, guidance_scale=None)
    curve = recovery_interval_coverage(points, sample_df)
    res = {"run_name": row["run_name"], "dataset_size": int(row["dataset_size"]), "k": k,
           "n_cosmologies": int(len(heldout)), "sample_path": str(path), "sample_sha256": sha256(path)}
    for name in names[:2]:
        s = rm.recovery_summary(points, name, rm.W_REAL_OMEGA_M if name == "Omega_m" else None)
        c95 = curve[(curve.parameter == name) & np.isclose(curve.nominal, 0.95)]
        s["coverage95"] = float(c95.empirical.iloc[0]) if len(c95) else float("nan")
        res.update({f"{name}_{kk}": v for kk, v in s.items()})
    # P(k) per cosmology against the real held-out slices; real split-half is the reference.
    binning = rm.pk_binning(128)
    real, _, real_sim, _ = load_heldout_real_slices(DATA_ROOT, heldout, args.heldout_slices_per_sim, probe_norm)
    real = rm.as_maps(real)
    pk_real, pk_gen = rm.power_spectra(real, binning), rm.power_spectra(samples, binning)
    nb = len(binning["kc"])
    real_mean = np.stack([pk_real[real_sim == s].mean(0) for s in heldout])
    ratios = pk_gen.reshape(len(heldout), k, nb).mean(1) / real_mean
    half = np.stack([pk_real[real_sim == s][::2].mean(0) / pk_real[real_sim == s][1::2].mean(0) for s in heldout])
    res.update(rm.pk_band_values(ratios, binning))
    res.update({f"ref_split_half_{kk}": v for kk, v in rm.pk_band_values(half, binning).items()})
    edges = rm.one_point_edges(real)
    op = rm.one_point_l1(samples, real, edges)
    res["one_point_l1"], res["one_point_outside_bins"] = op["l1"], op["outside_a"]
    res["ref_one_point_l1_split_half"] = rm.one_point_l1(real[::2], real[1::2], edges)["l1"]
    res.update({f"gen_{kk}": v for kk, v in rm.one_point_moments(samples).items()})
    res.update({f"heldout_{kk}": v for kk, v in rm.one_point_moments(real).items()})
    d2d = [rm.draw_to_draw_cosine(samples[i * k:(i + 1) * k])["mean"] for i in range(len(heldout))]
    d2d_real = [rm.draw_to_draw_cosine(real[real_sim == s])["mean"] for s in heldout]
    res["draw_to_draw_cosine_median"], res["ref_draw_to_draw_cosine_real_median"] = float(np.median(d2d)), float(np.median(d2d_real))
    if row.get("prepared_image_path") and not args.no_nearest:
        p = row["prepared_image_path"]
        raw_train = np.asarray(np.load(p if p.endswith(".npy") else p + "_images.npy", mmap_mode="r")[:, 0], np.float32)
        train = _normalize_reference_slices(raw_train, cfg, nk["center"], nk["xmax"])[:, 0]
        q, _, _ = balanced_draws(samples, heldout, k, min(k, args.nn_draws_per_cosmology))
        res["nearest_train_cosine_median"] = float(np.median(rm.plain_nearest_cosine(train, q)))
        res["ref_nearest_train_cosine_real_median"] = float(np.median(rm.plain_nearest_cosine(train, real[::8])))
    if out is not None:
        tag = row["run_name"]
        sample_df.to_csv(out / f"{tag}_per_sample_predictions.csv", index=False)
        points.to_csv(out / f"{tag}_per_cosmology_points.csv", index=False)
        curve.to_csv(out / f"{tag}_coverage_curve.csv", index=False)
        np.savez_compressed(out / f"{tag}_pk.npz", kc=binning["kc"], ratios=ratios, split_half=half,
                            draw_to_draw=np.asarray(d2d), heldout=heldout)
        plot_conditional(points, curve, ratios, half, binning, names[:2], out / f"{tag}_recovery_coverage_pk.png", tag)
    res["_points"] = points
    return res


def plot_conditional(points, curve, ratios, half, binning, names, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(names) + 2, figsize=(4.4 * (len(names) + 2), 4), constrained_layout=True)
    for ax, name in zip(axes, names):
        g = points[points.parameter == name].sort_values("theta_in")
        y = g.theta_rec_median.to_numpy()
        lo, hi = min(g.theta_in.min(), g.theta_rec_q16.min()), max(g.theta_in.max(), g.theta_rec_q84.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.errorbar(g.theta_in, y, yerr=[y - g.theta_rec_q16, g.theta_rec_q84 - y], fmt="o", ms=4, capsize=2,
                    color="#0072B2")
        s = rm.recovery_summary(points, name)
        ax.set(xlabel=f"true {name}", ylabel=f"recovered {name} (median, 16-84%)",
               title=f"{name}: median bias {s['median_bias']:+.3f}, cov68 {s['covered68']}/{s['n_cosmologies']}")
    ax = axes[len(names)]
    for name, color in zip(names, ["#0072B2", "#D55E00"]):
        c = curve[curve.parameter == name].sort_values("nominal")
        ax.plot(c.nominal, c.empirical, "o-", ms=3, color=color, label=name)
        ax.fill_between(c.nominal, c.ci_low, c.ci_high, color=color, alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="nominal interval", ylabel="empirical coverage (held-out cosmologies)",
           title="Coverage (recovery ensemble, not a posterior)")
    ax.legend(fontsize=8)
    ax = axes[-1]
    kc = binning["kc"]
    ax.plot(kc, np.exp(np.log(ratios).mean(0)), color="#0072B2", label="generated / real")
    ax.plot(kc, np.exp(np.log(half).mean(0)), "k:", label="real split-half")
    ax.axhline(1, color="0.6", lw=0.8, ls="--")
    ax.set(xscale="log", xlabel="k [grid units]", ylabel="P(k) ratio (geo. mean over cosmologies)", title="P(k)")
    ax.legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.2)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_conditional(args) -> None:
    from evaluate_nf_conditional_bias_probe import load_manifest, selected_rows

    out = fresh_dir(resolve(args.out_dir))
    trust = trust_status(resolve(args.regression_receipt))
    rows = selected_rows(load_manifest(ROOT, resolve(args.manifest)), args.run_name)
    if not rows:
        raise SystemExit("No runs selected.")
    enc, norm, names = load_probe(resolve(args.probe), args.device)
    results = []
    for row in rows:
        print(f"Scoring {row['run_name']}", flush=True)
        res = conditional_scores(row, args, enc, norm, names, out)
        res.pop("_points")
        results.append(res)
        print(json.dumps(json_ready({k: v for k, v in res.items() if "Omega_m" in k or k.startswith("pk_")})), flush=True)
    import pandas as pd

    pd.DataFrame(results).to_csv(out / "per_run.csv", index=False)
    write_json(out / "metrics.json", {**trust, "mode": "conditional", "W_omega_m": rm.W_REAL_OMEGA_M,
                                      "per_run_csv": str(out / "per_run.csv"), "runs": results})
    write_json(out / "provenance.json", {"git": git_rev(), "argv": sys.argv, "manifest": str(args.manifest),
                                         "manifest_sha256": sha256(resolve(args.manifest)), "probe": str(args.probe),
                                         "probe_sha256": sha256(resolve(args.probe)), **trust})
    print("Wrote", out, flush=True)


# ---------------------------------------------------------------- regression
def run_regression(args) -> None:
    import pandas as pd

    out = fresh_dir(resolve(args.out_dir))
    checks = []

    def check(name, got, want, tol, source):
        ok = bool(np.isfinite(got) and abs(got - want) <= tol)
        checks.append(dict(check=name, got=float(got), expected=float(want), tol=tol, passed=ok, source=source))
        print(f"{'PASS' if ok else 'FAIL'} {name}: got {got:.6f} expected {want:.6f} (tol {tol})", flush=True)

    # R1: PCA95 G for DiT-L16 (notebooks/dit_l16_zero_init_generalization_pca95.ipynb, cell 7 output; SAVE_OUTPUTS=False).
    src1 = "notebooks/dit_l16_zero_init_generalization_pca95.ipynb cell 7 output"
    expected = {64: (0.0, 0.916614), 1024: (0.486328, 0.998002), 2048: (0.970703, None)}
    runs = {}
    for plan in DIT_PLANS:
        for r in json.loads(Path(plan).read_text())["runs"]:
            if int(r["num_layers"]) == 16 and int(r["patch_size"]) == 8:
                s = Path(r["sample_path"].format(seed=123, sample_label="dpm50_n512"))
                if s.is_file() and s.with_suffix(".complete.json").is_file():
                    runs[int(r["dataset_size"])] = dict(config=r["config"], sample=s)
    largest = max(runs)
    fit, _ = load_pca_fit_maps(Path(runs[largest]["config"]), 1024)
    space = rm.PCASpace(fit, 32)
    del fit
    for n, (g_want, thr_want) in expected.items():
        from simdiff_eval.io import iter_real_reference_batches_from_config

        train = rm.as_maps(np.concatenate(list(iter_real_reference_batches_from_config(runs[n]["config"]))))
        nov = rm.pca_novelty(space, train, load_samples(runs[n]["sample"]), 0.95, "none")
        check(f"R1 DiT-L16 N={n} G", nov["G"], g_want, 2.5 / 512, src1)
        if thr_want is not None:
            check(f"R1 DiT-L16 N={n} threshold", nov["threshold"], thr_want, 1e-4, src1)

    # R3 and R4 share the continuous (Omega_m, sigma_8) manifest and the evaluation probe.
    from evaluate_nf_conditional_bias_probe import load_manifest

    manifest = {int(r["dataset_size"]): r for r in load_manifest(
        ROOT, ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json")}
    enc, norm, names = load_probe(resolve(args.probe), args.device)
    args.k, args.no_nearest = 64, True
    # R3: P(k) bands, N=256 (results/pk_bias_plots_20261006_161532/pk_ratio_summary.csv).
    src3 = "results/pk_bias_plots_20261006_161532/pk_ratio_summary.csv"
    pk_csv = pd.read_csv(ROOT / src3)
    res256 = conditional_scores(manifest[256], args, enc, norm, names, None)
    for band, label in [("large", "large scales"), ("mid", "mid scales"), ("small", "small scales")]:
        want = pk_csv[(pk_csv.model == "continuous (Ωm, σ8)") & (pk_csv.N == 256) & (pk_csv.band == label)]
        if len(want) != 1:
            checks.append(dict(check=f"R3 band {band}", passed=False, source=src3, note="row not found"))
            continue
        check(f"R3 continuous N=256 P(k) {band}", res256[f"pk_{band}"], float(want["P_gen/P_real"].iloc[0]), 1e-4, src3)
    # R4: probe recovery, N=64, per-cosmology medians against the saved CSV.
    src4 = "results/nf_conditional_omsig_continuous_200k/calibration_vgg/bias_probe_per_cosmology_points.csv"
    saved = pd.read_csv(ROOT / src4)
    saved = saved[(saved.dataset_size == 64) & (saved.guidance_label == "noguidance")]
    res64 = conditional_scores(manifest[64], args, enc, norm, names, None)
    pts = res64["_points"]
    merged = pts.merge(saved, on=["heldout_sim", "parameter"], suffixes=("", "_saved"))
    for name in ("Omega_m", "sigma_8"):
        m = merged[merged.parameter == name]
        check(f"R4 continuous N=64 {name} max |median - saved|",
              float(np.max(np.abs(m.theta_rec_median - m.theta_rec_median_saved))) if len(m) == 32 else np.nan,
              0.0, 2e-3, src4)
        want = rm.recovery_summary(saved, name)
        check(f"R4 continuous N=64 {name} median bias", res64[f"{name}_median_bias"], want["median_bias"], 2e-3, src4)
        check(f"R4 continuous N=64 {name} covered68", res64[f"{name}_covered68"], want["covered68"], 0, src4)

    passed = all(c["passed"] for c in checks)
    write_json(out / "regression_report.json", {"git": git_rev(), "argv": sys.argv, "passed": passed, "checks": checks,
                                                "not_checked": ["R2 training-theta report (no train-theta mode yet)"]})
    if passed:
        write_json(out / "regression_pass.json", {"git": git_rev(), "n_checks": len(checks),
                                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print("ALL PASSED" if passed else "REGRESSION FAILED", "->", out, flush=True)
    if not passed:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("unconditional", "conditional", "regression"), required=True)
    ap.add_argument("--out-dir", required=True, help="new directory; refused if it exists")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--probe", default=DEFAULT_PROBE)
    ap.add_argument("--regression-receipt", default=DEFAULT_RECEIPT)
    ap.add_argument("--run-name", action="append")
    # unconditional
    ap.add_argument("--preset", choices=("unet128_aug_sweep",))
    ap.add_argument("--runs-json")
    ap.add_argument("--dataset-size", type=int, action="append")
    ap.add_argument("--sample-label", default="dpm50")
    ap.add_argument("--pca-fit-config", default="local/nf_generalize_fig2/configs/nf_fig2_u128_d2p15_noaug_200k.yaml")
    ap.add_argument("--pca-components", type=int, default=32)
    ap.add_argument("--pca-fit-max-real", type=int, default=1024)
    ap.add_argument("--copy-quantile", type=float, default=0.95)
    ap.add_argument("--orbit-max-n", type=int, default=4096)
    ap.add_argument("--orbit-threshold-queries", type=int, default=2048)
    ap.add_argument("--heldout-raw-per-source", type=int, default=8)
    ap.add_argument("--max-heldout", type=int, default=512)
    ap.add_argument("--max-generated", type=int, default=512)
    ap.add_argument("--probe-max-train", type=int, default=1024)
    ap.add_argument("--no-probe", action="store_true")
    # conditional
    ap.add_argument("--manifest")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--heldout-slices-per-sim", type=int, default=128)
    ap.add_argument("--nn-draws-per-cosmology", type=int, default=16)
    ap.add_argument("--no-nearest", action="store_true")
    ap.add_argument("--probe-batch", type=int, default=32)
    args = ap.parse_args()
    global PROBE_BATCH
    PROBE_BATCH = args.probe_batch
    {"unconditional": run_unconditional, "conditional": run_conditional, "regression": run_regression}[args.mode](args)


if __name__ == "__main__":
    main()
