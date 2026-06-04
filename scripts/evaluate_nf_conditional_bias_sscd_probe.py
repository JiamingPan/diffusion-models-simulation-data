#!/usr/bin/env python
"""Evaluate continuous-cosmology calibration with the SSCD + Ridge encoder."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from evaluate_nf_conditional_bias_probe import (
    SWEEP_NAME,
    guidance_label,
    load_manifest,
    output_path_for,
    parse_guidance_scale,
    plot_calibration,
    selected_rows,
    slope_table,
)
from prepare_nf_conditional_u128_config import PARAM_NAMES
from simdiff_eval.sscd import load_sscd_torchscript, sscd_embeddings
from train_nf_conditional_bias_encoder import predict_ridge
from train_nf_conditional_bias_sscd_encoder import DEFAULT_ENCODER_PATH


@dataclass
class SSCDEncoder:
    model: object
    coef: np.ndarray
    intercept: np.ndarray
    param_mean: np.ndarray
    param_std: np.ndarray
    sscd_path: Path
    sscd_sha256: str
    image_size: int
    render_mode: str
    value_min: float
    value_max: float
    device: str
    feature_dim: int

    def predict_norm(self, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
        emb = sscd_embeddings(
            images,
            self.model,
            device=self.device,
            batch_size=batch_size,
            image_size=self.image_size,
            render_mode=self.render_mode,
            value_range=(self.value_min, self.value_max),
        )
        x = emb.numpy().astype(np.float32, copy=False)
        return predict_ridge(x, self.coef, self.intercept)

    def norm_to_raw(self, theta_norm: np.ndarray) -> np.ndarray:
        return theta_norm * self.param_std + self.param_mean


def load_sscd_encoder(project_dir: Path, encoder_path: Path, device: str | None) -> SSCDEncoder:
    with np.load(encoder_path, allow_pickle=True) as data:
        sscd_path = Path(str(data["sscd_path"].item())).expanduser()
        if not sscd_path.is_absolute():
            sscd_path = project_dir / sscd_path
        run_device = device or "auto"
        model = load_sscd_torchscript(sscd_path, device=run_device)
        return SSCDEncoder(
            model=model,
            coef=data["coef"].astype(np.float32),
            intercept=data["intercept"].astype(np.float32),
            param_mean=data["param_mean"].astype(np.float32),
            param_std=data["param_std"].astype(np.float32),
            sscd_path=sscd_path,
            sscd_sha256=str(data["sscd_sha256"].item()),
            image_size=int(data["sscd_image_size"]),
            render_mode=str(data["sscd_render_mode"].item()),
            value_min=float(data["sscd_value_min"]),
            value_max=float(data["sscd_value_max"]),
            device=run_device,
            feature_dim=int(data["feature_dim"]),
        )


def evaluate_run(
    *,
    project_dir: Path,
    row: dict,
    encoder: SSCDEncoder,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples-per-cosmology", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None)
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
    encoder = load_sscd_encoder(project_dir, encoder_path, args.device)
    output_dir = args.output_dir or project_dir / "results" / SWEEP_NAME / "sscd_calibration"
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
        "encoder_type": "sscd_ridge",
        "sscd_path": str(encoder.sscd_path),
        "sscd_sha256": encoder.sscd_sha256,
        "sscd_image_size": encoder.image_size,
        "sscd_render_mode": encoder.render_mode,
        "sscd_value_range": [encoder.value_min, encoder.value_max],
        "sscd_feature_dim": encoder.feature_dim,
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
