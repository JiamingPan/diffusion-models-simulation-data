#!/usr/bin/env python
"""Write the metrics JSONs behind the 2026-10-09 ledger results (same definitions as
notebooks/results_review_20261009.ipynb). Each output is a new file; existing files are never overwritten."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))
from simdiff_eval import run_metrics as rm  # noqa: E402
from simdiff_eval.conditional_unet_diagnostics import nearest_centered_pixel_cosine, normalize_raw_hi  # noqa: E402

W = rm.W_REAL_OMEGA_M
HELDOUT = np.arange(900, 932)
NOTEBOOK = "notebooks/results_review_20261009.ipynb"
OUT_NAME = "ledger_metrics_20261009.json"


def write(folder: str, obj: dict) -> Path:
    path = ROOT / folder / OUT_NAME
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")
    print("wrote", path)
    return path


def recovery(folder: str, run_name: str, param: str = "Omega_m") -> dict:
    pts = pd.read_csv(ROOT / folder / "bias_probe_per_cosmology_points.csv")
    pts = pts[(pts.run_name == run_name) & (pts.guidance_label == "noguidance")]
    assert set(pts.heldout_sim) == set(HELDOUT), (folder, run_name)
    pred = pd.read_csv(ROOT / folder / "bias_probe_per_sample_predictions.csv")
    d = pred[(pred.run_name == run_name) & (pred.guidance_label == "noguidance") & (pred.parameter == param)]
    d = d.sort_values(["heldout_sim", "seed_index"])
    draws = d.theta_rec.to_numpy().reshape(32, -1)
    truth = d.groupby("heldout_sim").theta_in.first().loc[HELDOUT].to_numpy()
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=1)
    s = rm.recovery_summary(pts, param, W if param == "Omega_m" else None)
    s["covered95"] = int(((lo <= truth) & (truth <= hi)).sum())
    s["source"] = f"{folder}/bias_probe_per_cosmology_points.csv [{run_name}]"
    return s


_REAL = {}


def map_stats(sample_path: str, train_raw_path: str) -> dict:
    from simdiff_eval.probe_eval import load_heldout_real_slices
    from prepare_nf_conditional_u128_config import DATA_ROOT

    norm = dict(np.load(ROOT / "results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz",
                        allow_pickle=True)["normalization"].item())
    binning = rm.pk_binning(128)
    if not _REAL:
        real, _, sim, _ = load_heldout_real_slices(DATA_ROOT, HELDOUT, 128, norm)
        pk = rm.power_spectra(rm.as_maps(real), binning)
        _REAL["mean"] = np.stack([pk[sim == s].mean(0) for s in HELDOUT])
    with np.load(ROOT / sample_path, allow_pickle=True) as z:
        gen = rm.as_maps(z["samples"])
        k = int(z["samples_per_cosmology"])
        assert np.array_equal(z["heldout_indices"], HELDOUT)
    ratios = rm.power_spectra(gen, binning).reshape(32, k, -1).mean(1) / _REAL["mean"]
    out = {key: v for key, v in rm.pk_band_values(ratios, binning).items() if not key.endswith("pct")}
    train = normalize_raw_hi(np.load(train_raw_path, mmap_mode="r")[:, 0], norm)
    cos, j = nearest_centered_pixel_cosine(gen, train)
    out["pix_std_ratio_median"] = float(np.median(gen.reshape(len(gen), -1).std(1) / train.reshape(len(train), -1)[j].std(1)))
    out["nearest_train_cosine_median"] = float(np.median(cos))
    out["sample_path"] = sample_path
    return out


def condmech() -> Path:
    cm = json.loads((ROOT / "local/nf_conditioning_mechanism_n256/manifest.json").read_text())
    base = next(r for r in json.loads((ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json").read_text())
                if r["dataset_size"] == 256)
    runs = [("baseline_cross_attn", "results/nf_conditional_omsig_continuous_200k/calibration_vgg", base)] + \
           [(r["run_name"].split("n256_")[1], "results/nf_conditioning_mechanism_n256/calibration_vgg", r) for r in cm]
    arms = {}
    for label, folder, row in runs:
        arms[label] = {"run_name": row["run_name"], "Omega_m": recovery(folder, row["run_name"]),
                       "sigma_8": recovery(folder, row["run_name"], "sigma_8"),
                       **map_stats(row["sample_path"].format(seed=123, k=64), row["prepared_image_path"])}
    b = arms["baseline_cross_attn"]
    decision = {}
    for label, a in arms.items():
        if label == "baseline_cross_attn":
            continue
        change = {band: abs(a[f"pk_{band}"] - 1) - abs(b[f"pk_{band}"] - 1) for band in ("large", "mid", "small")}
        decision[label] = {"abs_Omega_m_bias_better": abs(a["Omega_m"]["median_bias"]) < abs(b["Omega_m"]["median_bias"]),
                           "pk_band_change_vs_baseline": change,
                           "adopt": abs(a["Omega_m"]["median_bias"]) < abs(b["Omega_m"]["median_bias"])
                           and all(c < 0 for c in change.values()) and all(c <= 0.03 for c in change.values())}
    return write("results/nf_conditioning_mechanism_n256", {
        "ledger_run": "nf_condmech_n256", "notebook": NOTEBOOK, "W": W, "arms": arms, "decision_rule": decision,
        "pk_band_definition": "exp(mean over 32 cosmologies of mean log P_gen/P_real in band); large kc<8, mid 8-24, small >=24"})


def om_n1024() -> Path:
    om = {r["dataset_size"]: r for r in json.loads((ROOT / "local/nf_conditional_om_continuous_200k/manifest.json").read_text())}
    os_ = {r["dataset_size"]: r for r in json.loads((ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json").read_text())}
    out = {}
    for n in (64, 256, 1024):
        folder = "results/nf_conditional_om_continuous_200k/" + ("calibration_vgg_n1024" if n == 1024 else "calibration_vgg_n64_n256")
        out[f"Omega_m_only_N{n}"] = {**recovery(folder, om[n]["run_name"]),
                                     **map_stats(om[n]["sample_path"].format(seed=123, k=64), om[n]["prepared_image_path"])}
        out[f"Omega_m_sigma_8_N{n}"] = {**recovery("results/nf_conditional_omsig_continuous_200k/calibration_vgg", os_[n]["run_name"]),
                                        **map_stats(os_[n]["sample_path"].format(seed=123, k=64), os_[n]["prepared_image_path"])}
    r = out["Omega_m_only_N1024"]
    out["prediction_check"] = {"abs_bias_lt_0p25W": abs(r["median_bias"]) < 0.25 * W, "covered68_ge_16": r["covered68"] >= 16,
                               "falsified": abs(r["median_bias"]) >= 0.25 * W or r["covered68"] <= 12}
    return write("results/nf_conditional_om_continuous_200k/calibration_vgg_n1024",
                 {"ledger_run": "nf_om_continuous_n1024_sample_eval", "notebook": NOTEBOOK, "W": W, **out})


def om_n64_n256() -> Path:
    om = {r["dataset_size"]: r for r in json.loads((ROOT / "local/nf_conditional_om_continuous_200k/manifest.json").read_text())}
    folder = "results/nf_conditional_om_continuous_200k/calibration_vgg_n64_n256"
    return write(folder, {"ledger_run": "nf_om_continuous_eval_n64_n256", "notebook": NOTEBOOK, "W": W,
                          **{f"Omega_m_only_N{n}": recovery(folder, om[n]["run_name"]) for n in (64, 256)}})


def ckpt() -> Path:
    a = ROOT / "results/unet128_aug_sweep/evaluate_run_ckpt_v1"
    ck = pd.read_csv(a / "per_run.csv")
    ck = ck[ck.status == "ok"].copy()
    ck["update_k"] = ck.run_name.str.extract(r"_u(\d+)k$").astype(int)
    ck["heldout_orbit_nn_median"] = [float(np.median(np.load(a / f"{r}_per_sample.npz")["heldout_orbit_nn"])) for r in ck.run_name]
    s = ck[ck.arm == "d4shift"]
    e = ck[(ck.arm == "d4equiv") & (ck.dataset_size == 64)].sort_values("update_k")
    table = {f"{arm}_N{n}": dict(zip(g.sort_values("update_k").update_k.astype(int).astype(str),
                                     g.sort_values("update_k").orbit_copy_fraction.round(4)))
             for (arm, n), g in ck.groupby(["arm", "dataset_size"])}
    return write("results/unet128_aug_sweep/evaluate_run_ckpt_v1", {
        "ledger_run": "unet128_aug_ckpt_sweep", "notebook": NOTEBOOK, "source": str(a / "per_run.csv"),
        "orbit_copy_fraction_by_update_k": table,
        "d4shift_max_copy_fraction": float(s.orbit_copy_fraction.max()),
        "d4shift_max_abs_orbit_nn_minus_heldout_median": float((s.orbit_nn_median - s.heldout_orbit_nn_median).abs().max()),
        "d4equiv_N64_copy_fraction_first_last": [float(e.orbit_copy_fraction.iloc[0]), float(e.orbit_copy_fraction.iloc[-1])]})


def patch() -> Path:
    a = ROOT / "results/unet128_aug_sweep/patch_mosaic_v1"
    pm = pd.read_csv(a / "per_run.csv")
    pm = pm[pm.status == "ok"]
    cols = [f"patch{p}_{c}" for p in (8, 16, 32, 64) for c in ("excess", "excess_ci_low", "excess_ci_high", "copy_fraction")]
    rows = {f"{r.arm}_N{r.dataset_size}": {c: float(r[c]) for c in cols} for _, r in pm.iterrows()}
    na, ds = rows["noaug_N64"], rows["d4shift_N64"]
    seq = [rows[f"d4shift_N{n}"]["patch16_excess"] for n in (64, 128, 256, 1024)]
    checks = {"1_sanity_noaug_N64": all(na[f"patch{p}_excess"] > 0 for p in (8, 16, 32, 64)) and na["patch64_copy_fraction"] >= 0.9,
              "2_d4shift_N64_small_p_excess_ci_excludes_0": all(ds[f"patch{p}_excess_ci_low"] > 0 for p in (8, 16)),
              "3_d4shift_p16_excess_decreasing_in_N": bool(np.all(np.diff(seq) < 0))}
    return write("results/unet128_aug_sweep/patch_mosaic_v1", {
        "ledger_run": "unet128_aug_patch_mosaic", "notebook": NOTEBOOK, "source": str(a / "per_run.csv"),
        "runs": rows, "checks": checks, "d4shift_patch16_excess_N64_128_256_1024": seq})


def sampling() -> Path:
    runs = json.loads((ROOT / "local/unet128_aug_sweep/runs.json").read_text())
    files = {}
    for r in runs:
        p = ROOT / r["sample_path"].format(seed=123)
        with np.load(p) as z:
            x = z["samples"]
            files[r["run_name"]] = {"shape": list(x.shape), "finite": bool(np.isfinite(x).all())}
    ok = all(f["shape"] == [512, 1, 128, 128] and f["finite"] for f in files.values())
    return write("results/unet128_aug_sweep/samples", {"ledger_run": "unet128_aug_sweep_sample", "n_files": len(files),
                                                        "all_512x1x128x128_finite": ok, "files": files})


if __name__ == "__main__":
    steps = {f.__name__: f for f in (condmech, om_n1024, om_n64_n256, ckpt, patch, sampling)}
    for name in sys.argv[1:] or steps:
        steps[name]()
