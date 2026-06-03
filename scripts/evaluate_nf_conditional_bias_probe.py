#!/usr/bin/env python
"""Evaluate continuous-cosmology calibration bias from generated HI fields."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_nf_conditional_u128_config import PARAM_NAMES
from train_nf_conditional_bias_encoder import FrozenPCA, as_nchw, load_pca, predict_ridge


SWEEP_NAME = "nf_conditional_bias_probe"
DEFAULT_ENCODER_PATH = "results/nf_conditional_bias_probe/encoder/frozen_pca_ridge_encoder.npz"


def parse_guidance_scale(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if cleaned in {"", "none", "null", "noguidance", "no_guidance"}:
        return None
    return float(cleaned)


def guidance_label(guidance_scale: float | None) -> str:
    if guidance_scale is None:
        return "noguidance"
    return f"g{float(guidance_scale):.3f}".rstrip("0").rstrip(".").replace(".", "p")


@dataclass
class Encoder:
    pca: FrozenPCA
    coef: np.ndarray
    intercept: np.ndarray
    param_mean: np.ndarray
    param_std: np.ndarray
    pca_basis_path: Path
    pca_basis_sha256: str

    def predict_norm(self, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
        z = self.pca.transform(as_nchw(images), batch_size=batch_size)
        return predict_ridge(z, self.coef, self.intercept)

    def norm_to_raw(self, theta_norm: np.ndarray) -> np.ndarray:
        return theta_norm * self.param_std + self.param_mean


def load_manifest(project_dir: Path, manifest_path: Path | None) -> list[dict[str, Any]]:
    path = manifest_path or project_dir / "local" / SWEEP_NAME / "manifest.json"
    with path.open() as f:
        return json.load(f)


def selected_rows(rows: list[dict[str, Any]], run_names: list[str] | None) -> list[dict[str, Any]]:
    if run_names:
        wanted = set(run_names)
        rows = [row for row in rows if row["run_name"] in wanted]
    return sorted(rows, key=lambda row: int(row["dataset_size"]))


def output_path_for(project_dir: Path, row: dict[str, Any], seed: int, k: int, guidance_scale: float | None) -> Path:
    raw = str(row["sample_path"])
    label = guidance_label(guidance_scale)
    try:
        rel = raw.format(seed=seed, sample_label="dpm50", k=k, guidance=label)
    except KeyError:
        rel = raw.format(seed=seed, sample_label="dpm50", k=k)
    path = project_dir / rel
    if guidance_scale is not None and "{guidance}" not in raw:
        path = path.with_name(f"{path.stem}_{label}{path.suffix}")
    return path


def load_encoder(project_dir: Path, encoder_path: Path) -> Encoder:
    with np.load(encoder_path, allow_pickle=True) as data:
        basis_path = Path(str(data["pca_basis_path"].item()))
        if not basis_path.is_absolute():
            basis_path = project_dir / basis_path
        pca = load_pca(basis_path)
        return Encoder(
            pca=pca,
            coef=data["coef"].astype(np.float32),
            intercept=data["intercept"].astype(np.float32),
            param_mean=data["param_mean"].astype(np.float32),
            param_std=data["param_std"].astype(np.float32),
            pca_basis_path=basis_path,
            pca_basis_sha256=str(data["pca_basis_sha256"].item()),
        )


def fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or float(np.var(x)) <= 1.0e-30:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(slope), float(intercept)


def bootstrap_slope_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    slope, _ = fit_slope(x, y)
    if len(x) < 2:
        return slope, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(x), size=len(x))
        boot_slope, _ = fit_slope(x[idx], y[idx])
        if np.isfinite(boot_slope):
            vals.append(boot_slope)
    if not vals:
        return slope, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.16, 0.84])
    return slope, float(lo), float(hi)


def evaluate_run(
    *,
    project_dir: Path,
    row: dict[str, Any],
    encoder: Encoder,
    seed: int,
    k: int,
    embedding_batch_size: int,
    guidance_scale: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_path = output_path_for(project_dir, row, seed, k, guidance_scale)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    with np.load(sample_path, allow_pickle=True) as data:
        samples = data["samples"].astype(np.float32)
        theta_raw = data["theta_raw"].astype(np.float32)
        heldout_indices = data["heldout_indices"].astype(np.int64)
        k_file = int(data["samples_per_cosmology"])
    if k_file != k:
        raise ValueError(f"{sample_path} has k={k_file}, requested k={k}.")
    if len(samples) != len(theta_raw) * k:
        raise ValueError(f"Expected {len(theta_raw) * k} samples, got {len(samples)}.")

    theta_pred_norm = encoder.predict_norm(samples, batch_size=embedding_batch_size)
    theta_pred_raw = encoder.norm_to_raw(theta_pred_norm)
    sample_rows = []
    point_rows = []
    for h, sim_idx in enumerate(heldout_indices):
        sl = slice(h * k, (h + 1) * k)
        for s in range(k):
            rec = theta_pred_raw[sl.start + s]
            for p, name in enumerate(PARAM_NAMES):
                sample_rows.append(
                    {
                        "run_name": row["run_name"],
                        "regime": row["regime"],
                        "dataset_size": int(row["dataset_size"]),
                        "cfg_dropout": float(row.get("cfg_dropout", 0.0)),
                        "guidance_scale": np.nan if guidance_scale is None else float(guidance_scale),
                        "guidance_label": guidance_label(guidance_scale),
                        "heldout_sim": int(sim_idx),
                        "seed_index": int(s),
                        "parameter": name,
                        "theta_in": float(theta_raw[h, p]),
                        "theta_rec": float(rec[p]),
                    }
                )
        rec_block = theta_pred_raw[sl]
        for p, name in enumerate(PARAM_NAMES):
            vals = rec_block[:, p]
            q16, med, q84 = np.quantile(vals, [0.16, 0.5, 0.84])
            point_rows.append(
                {
                    "run_name": row["run_name"],
                    "regime": row["regime"],
                    "dataset_size": int(row["dataset_size"]),
                    "cfg_dropout": float(row.get("cfg_dropout", 0.0)),
                    "guidance_scale": np.nan if guidance_scale is None else float(guidance_scale),
                    "guidance_label": guidance_label(guidance_scale),
                    "heldout_sim": int(sim_idx),
                    "parameter": name,
                    "theta_in": float(theta_raw[h, p]),
                    "theta_rec_median": float(med),
                    "theta_rec_q16": float(q16),
                    "theta_rec_q84": float(q84),
                    "theta_rec_std": float(np.std(vals)),
                    "n_samples": int(k),
                }
            )
    return pd.DataFrame(sample_rows), pd.DataFrame(point_rows)


def slope_table(points: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for (guidance, cfg_dropout, regime, run_name, dataset_size, parameter), sub in points.groupby(
        ["guidance_label", "cfg_dropout", "regime", "run_name", "dataset_size", "parameter"], sort=False
    ):
        x = sub["theta_in"].to_numpy(float)
        y = sub["theta_rec_median"].to_numpy(float)
        slope, lo, hi = bootstrap_slope_ci(x, y, n_boot=n_boot, seed=seed)
        _, intercept = fit_slope(x, y)
        rows.append(
            {
                "regime": regime,
                "run_name": run_name,
                "dataset_size": int(dataset_size),
                "cfg_dropout": float(cfg_dropout),
                "guidance_label": guidance,
                "parameter": parameter,
                "slope": slope,
                "slope_ci16": lo,
                "slope_ci84": hi,
                "intercept": intercept,
                "n_heldout": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration(points: pd.DataFrame, slopes: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    colors = {"memorization": "#d62728", "generalization": "#1f77b4"}
    markers = {"memorization": "o", "generalization": "s"}
    regime_names = {"memorization": "memorization", "generalization": "generalization"}
    show_cfg = bool(points["cfg_dropout"].nunique() > 1 or float(points["cfg_dropout"].iloc[0]) != 0.0)
    show_guidance = bool(points["guidance_label"].nunique() > 1 or points["guidance_label"].iloc[0] != "noguidance")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for ax, param in zip(axes.ravel(), PARAM_NAMES):
        sub_param = points[points["parameter"] == param]
        if sub_param.empty:
            ax.set_visible(False)
            continue
        lo = float(min(sub_param["theta_in"].min(), sub_param["theta_rec_q16"].min()))
        hi = float(max(sub_param["theta_in"].max(), sub_param["theta_rec_q84"].max()))
        pad = 0.06 * max(hi - lo, 1.0e-6)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.6, ls="--", alpha=0.75, label="ideal")
        for (regime, run_name, cfg_dropout), sub in sub_param.groupby(["regime", "run_name", "cfg_dropout"], sort=False):
            color = colors.get(regime, "#333333")
            label = regime_names.get(regime, regime)
            if show_cfg:
                label += f" cfg={float(cfg_dropout):g}"
            if show_guidance:
                label += f" {sub['guidance_label'].iloc[0]}"
            y = sub["theta_rec_median"].to_numpy(float)
            yerr = np.vstack([
                y - sub["theta_rec_q16"].to_numpy(float),
                sub["theta_rec_q84"].to_numpy(float) - y,
            ])
            ax.errorbar(
                sub["theta_in"],
                y,
                yerr=yerr,
                fmt=markers.get(regime, "o"),
                ms=7.0,
                lw=1.8,
                capsize=3.0,
                color=color,
                alpha=0.86,
                label=label,
            )
            slope_row = slopes[
                (slopes["parameter"] == param)
                & (slopes["regime"] == regime)
                & (slopes["run_name"] == run_name)
            ]
            if not slope_row.empty:
                slope = float(slope_row["slope"].iloc[0])
                intercept = float(slope_row["intercept"].iloc[0])
                ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color=color, lw=2.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(param, fontsize=15)
        ax.set_xlabel(r"input $\theta$")
        ax.set_ylabel(r"recovered $\theta$")
        ax.tick_params(labelsize=12)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=13)
    fig.suptitle("Continuous HI cosmology calibration: recovered vs input parameters", y=1.08, fontsize=22)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples-per-cosmology", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--guidance-scale",
        action="append",
        help="Optional CFG guidance scale to evaluate. May be repeated; use 'none' for no guidance.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    rows = selected_rows(load_manifest(project_dir, args.manifest), args.run_name)
    if not rows:
        raise SystemExit("No runs selected.")
    encoder_path = project_dir / args.encoder
    encoder = load_encoder(project_dir, encoder_path)
    output_dir = args.output_dir or project_dir / "results" / SWEEP_NAME / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    guidance_scales = [parse_guidance_scale(x) for x in args.guidance_scale] if args.guidance_scale else [None]

    sample_tables = []
    point_tables = []
    for row in rows:
        k = int(args.samples_per_cosmology or row.get("heldout_samples_per_cosmology", 64))
        for guidance_scale in guidance_scales:
            sample_df, point_df = evaluate_run(
                project_dir=project_dir,
                row=row,
                encoder=encoder,
                seed=args.seed,
                k=k,
                embedding_batch_size=args.embedding_batch_size,
                guidance_scale=guidance_scale,
            )
            sample_tables.append(sample_df)
            point_tables.append(point_df)

    samples = pd.concat(sample_tables, ignore_index=True)
    points = pd.concat(point_tables, ignore_index=True)
    slopes = slope_table(points, n_boot=args.bootstrap, seed=args.seed)

    samples_path = output_dir / "bias_probe_per_sample_predictions.csv"
    points_path = output_dir / "bias_probe_per_cosmology_points.csv"
    slopes_path = output_dir / "bias_probe_regime_slopes.csv"
    samples.to_csv(samples_path, index=False)
    points.to_csv(points_path, index=False)
    slopes.to_csv(slopes_path, index=False)
    if points["guidance_label"].nunique() == 1:
        plot_calibration(points, slopes, output_dir / "bias_probe_calibration_recovered_vs_input.png")
    else:
        for label, sub_points in points.groupby("guidance_label", sort=False):
            sub_slopes = slopes[slopes["guidance_label"] == label]
            plot_calibration(
                sub_points,
                sub_slopes,
                output_dir / f"bias_probe_calibration_recovered_vs_input_{label}.png",
            )

    metadata = {
        "encoder_path": str(encoder_path),
        "pca_basis_path": str(encoder.pca_basis_path),
        "pca_basis_sha256": encoder.pca_basis_sha256,
        "pca_rank": encoder.pca.rank,
        "pca_explained_variance_sum": encoder.pca.explained_variance_sum,
        "param_names": PARAM_NAMES,
        "guidance_labels": sorted(points["guidance_label"].unique().tolist()),
    }
    (output_dir / "bias_probe_eval_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {samples_path}")
    print(f"Wrote {points_path}")
    print(f"Wrote {slopes_path}")
    print(f"Wrote {output_dir / 'bias_probe_calibration_recovered_vs_input.png'}")
    print(slopes[slopes["parameter"].isin(["Omega_m", "sigma_8"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
