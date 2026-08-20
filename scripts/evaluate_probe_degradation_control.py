#!/usr/bin/env python
"""Evaluate generated-power-matched degraded real maps with a frozen VGG probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simdiff_eval.metrics import field_histogram  # noqa: E402
from simdiff_eval.probe_controls import (  # noqa: E402
    C4_LIMITATION,
    TransformSpec,
    aggregate_prediction_table,
    build_run_manifest,
    deterministic_cosmology_split,
    evaluate_transform_specs,
    fit_gaussian_smoothing,
    power_ratio_transfer,
    subset_generated_cosmologies,
)
from simdiff_eval.probe_transforms import get_transform, transfer_transform  # noqa: E402


EXPECTED_HELDOUT = np.arange(900, 932, dtype=np.int64)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _json_array(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float64).astype(float).tolist()


def _prediction_table(
    *,
    images: np.ndarray,
    theta_raw: np.ndarray,
    sim_index: np.ndarray,
    z_index: np.ndarray,
    encoder: Any,
    batch_size: int,
    transform_name: str,
    transform_family: str,
    source: str,
    run_name: str,
    dataset_size: int | None,
) -> pd.DataFrame:
    table = evaluate_transform_specs(
        images,
        theta_raw,
        sim_index,
        z_index,
        encoder,
        [TransformSpec("identity", "identity", get_transform("identity"))],
        batch_size=int(batch_size),
    )
    table["transform"] = transform_name
    table["transform_family"] = transform_family
    table["source"] = source
    table["run_name"] = run_name
    table["dataset_size"] = dataset_size
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run degraded real-map controls with an existing frozen VGG probe."
    )
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--data-root")
    parser.add_argument(
        "--encoder",
        default="results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples-per-cosmology", type=int)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--pk-nbins", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from evaluate_nf_conditional_bias_probe import (
        load_manifest,
        load_vgg_encoder,
        output_path_for,
        selected_rows,
    )
    from prepare_nf_conditional_u128_config import DATA_ROOT
    from simdiff_eval.probe_eval import load_heldout_real_slices

    project_dir = Path(args.project_dir).resolve()
    data_root = args.data_root or DATA_ROOT
    encoder_path = Path(args.encoder)
    if not encoder_path.is_absolute():
        encoder_path = project_dir / encoder_path
    output_dir = args.output_dir or (
        project_dir / "results" / "nf_conditional_bias_probe" / "degradation_controls"
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = selected_rows(load_manifest(project_dir, args.manifest), args.run_name)
    if not rows:
        raise SystemExit("No generated runs selected")
    with np.load(encoder_path, allow_pickle=True) as data:
        normalization = data["normalization"].item()
        heldout = data["heldout_indices"].astype(np.int64)
    if not np.array_equal(heldout, EXPECTED_HELDOUT):
        raise ValueError(
            f"Encoder held-out indices must be 900..931; found {heldout.tolist()}"
        )

    real_images, real_theta, real_sim, real_z = load_heldout_real_slices(
        data_root,
        heldout,
        slices_per_sim=128,
        norm=normalization,
    )
    derivation_sims, evaluation_sims = deterministic_cosmology_split(
        heldout,
        seed=args.split_seed,
    )
    derivation_mask = np.isin(real_sim, derivation_sims)
    evaluation_mask = np.isin(real_sim, evaluation_sims)
    real_derivation = real_images[derivation_mask]
    real_evaluation = real_images[evaluation_mask]
    theta_evaluation = real_theta[evaluation_mask]
    sim_evaluation = real_sim[evaluation_mask]
    z_evaluation = real_z[evaluation_mask]

    encoder = load_vgg_encoder(project_dir, encoder_path, args.device)
    prediction_tables = [
        _prediction_table(
            images=real_evaluation,
            theta_raw=theta_evaluation,
            sim_index=sim_evaluation,
            z_index=z_evaluation,
            encoder=encoder,
            batch_size=args.embedding_batch_size,
            transform_name="identity",
            transform_family="identity",
            source="real_original",
            run_name="real_original",
            dataset_size=None,
        )
    ]
    histogram_payload: dict[str, Any] = {
        "limitation": C4_LIMITATION,
        "real_original": field_histogram(real_evaluation),
        "runs": {},
    }
    power_payload: dict[str, Any] = {
        "limitation": C4_LIMITATION,
        "pk_nbins": int(args.pk_nbins),
        "runs": {},
    }
    transform_records: list[dict[str, Any]] = [
        {"name": "identity", "family": "identity"}
    ]

    for row in rows:
        sample_count = int(
            args.samples_per_cosmology
            or row.get("heldout_samples_per_cosmology", 64)
        )
        sample_path = output_path_for(
            project_dir,
            row,
            args.seed,
            sample_count,
            None,
        )
        with np.load(sample_path, allow_pickle=True) as data:
            generated = data["samples"].astype(np.float32)
            generated_theta = data["theta_raw"].astype(np.float32)
            generated_heldout = data["heldout_indices"].astype(np.int64)
            stored_count = int(data["samples_per_cosmology"])
        if stored_count != sample_count:
            raise ValueError(
                f"{sample_path} has {stored_count} samples per cosmology; expected {sample_count}"
            )
        if not np.array_equal(generated_heldout, heldout):
            raise ValueError(f"Generated held-out ordering differs in {sample_path}")

        generated_derivation, _, _, _ = subset_generated_cosmologies(
            generated,
            generated_theta,
            generated_heldout,
            samples_per_cosmology=sample_count,
            selected_simulations=derivation_sims,
        )
        generated_evaluation, generated_eval_theta, generated_eval_sim, generated_eval_index = (
            subset_generated_cosmologies(
                generated,
                generated_theta,
                generated_heldout,
                samples_per_cosmology=sample_count,
                selected_simulations=evaluation_sims,
            )
        )
        k_bins, real_mean, generated_mean, ratio, measured_transfer = power_ratio_transfer(
            real_derivation,
            generated_derivation,
            nbins=args.pk_nbins,
        )
        sigma = fit_gaussian_smoothing(k_bins, ratio)
        gaussian_transfer = np.exp(-0.5 * (sigma * k_bins) ** 2)
        measured_images, _ = transfer_transform(k_bins, measured_transfer)(real_evaluation)
        gaussian_images, _ = transfer_transform(k_bins, gaussian_transfer)(real_evaluation)

        dataset_size = int(row["dataset_size"])
        run_name = str(row["run_name"])
        measured_name = f"transfer_Tk_N{dataset_size}"
        gaussian_name = f"gaussian_smoothing_N{dataset_size}"
        generated_name = f"generated_N{dataset_size}"
        prediction_tables.extend(
            [
                _prediction_table(
                    images=measured_images,
                    theta_raw=theta_evaluation,
                    sim_index=sim_evaluation,
                    z_index=z_evaluation,
                    encoder=encoder,
                    batch_size=args.embedding_batch_size,
                    transform_name=measured_name,
                    transform_family="transfer",
                    source="real_measured_transfer",
                    run_name=run_name,
                    dataset_size=dataset_size,
                ),
                _prediction_table(
                    images=gaussian_images,
                    theta_raw=theta_evaluation,
                    sim_index=sim_evaluation,
                    z_index=z_evaluation,
                    encoder=encoder,
                    batch_size=args.embedding_batch_size,
                    transform_name=gaussian_name,
                    transform_family="gaussian",
                    source="real_gaussian",
                    run_name=run_name,
                    dataset_size=dataset_size,
                ),
                _prediction_table(
                    images=generated_evaluation,
                    theta_raw=generated_eval_theta,
                    sim_index=generated_eval_sim,
                    z_index=generated_eval_index,
                    encoder=encoder,
                    batch_size=args.embedding_batch_size,
                    transform_name=generated_name,
                    transform_family="generated",
                    source="generated",
                    run_name=run_name,
                    dataset_size=dataset_size,
                ),
            ]
        )
        transform_records.extend(
            [
                {
                    "name": measured_name,
                    "family": "transfer",
                    "dataset_size": dataset_size,
                    "k_bins": _json_array(k_bins),
                    "transfer_values": _json_array(measured_transfer),
                },
                {
                    "name": gaussian_name,
                    "family": "gaussian",
                    "dataset_size": dataset_size,
                    "sigma_pixels": float(sigma),
                },
                {
                    "name": generated_name,
                    "family": "generated",
                    "dataset_size": dataset_size,
                },
            ]
        )
        power_payload["runs"][run_name] = {
            "dataset_size": dataset_size,
            "sample_path": str(sample_path.resolve()),
            "k_bins": _json_array(k_bins),
            "real_power_mean": _json_array(real_mean),
            "generated_power_mean": _json_array(generated_mean),
            "power_ratio": _json_array(ratio),
            "measured_transfer": _json_array(measured_transfer),
            "gaussian_sigma_pixels": float(sigma),
            "gaussian_transfer": _json_array(gaussian_transfer),
        }
        histogram_payload["runs"][run_name] = {
            "dataset_size": dataset_size,
            "real_measured_transfer": field_histogram(measured_images),
            "real_gaussian": field_histogram(gaussian_images),
            "generated": field_histogram(generated_evaluation),
        }

    predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics = aggregate_prediction_table(
        predictions,
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    transform_metadata = (
        predictions[["transform", "source", "run_name", "dataset_size"]]
        .drop_duplicates(subset=["transform"])
        .set_index("transform")
        .to_dict("index")
    )
    for metric in metrics["metrics"]:
        metric.update(transform_metadata[metric["transform"]])
    metrics["limitation"] = C4_LIMITATION

    manifest = build_run_manifest(
        project_dir=project_dir,
        encoder_path=encoder_path,
        head_path=encoder.model_path,
        heldout_indices=heldout,
        slices_per_sim=128,
        transforms=transform_records,
        seeds={
            "sample": int(args.seed),
            "bootstrap": int(args.bootstrap_seed),
            "split": int(args.split_seed),
        },
        arguments=vars(args),
        extra={
            "derivation_simulations": derivation_sims,
            "evaluation_simulations": evaluation_sims,
            "limitation": C4_LIMITATION,
        },
    )

    predictions_path = output_dir / "probe_degradation_predictions.csv"
    metrics_path = output_dir / "probe_degradation_metrics.json"
    power_path = output_dir / "power_transfer_curves.json"
    histogram_path = output_dir / "field_histograms.json"
    manifest_path = output_dir / "manifest.json"
    _write_csv(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    _write_json(power_path, power_payload)
    _write_json(histogram_path, histogram_payload)
    _write_json(manifest_path, manifest)
    for path in (
        predictions_path,
        metrics_path,
        power_path,
        histogram_path,
        manifest_path,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
