#!/usr/bin/env python
"""Measure exact-subset fidelity and patch artifacts for the DiT-L16 sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import yaml

PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from simdiff_eval.dit_diagnostics import (
    bootstrap_histogram_l1_interval,
    bootstrap_power_log10_mae_interval,
    patch_boundary_per_image,
    patch_boundary_statistics,
    selected_power_bin_statistics,
    split_reference_physics_floor,
    summarize_patch_boundary_series,
    two_sample_selected_power_ratio_statistics,
)
from simdiff_eval.io import iter_real_reference_batches_from_config
from simdiff_eval.metrics import (
    PHYSICAL_HIST_EDGES,
    batch_power_spectra,
    histogram_probability_and_coverage,
)
from validate_nf_generalize_fig2_dit_sample import validate_sample_file


SWEEP = "nf_generalize_fig2_dit_l16_continue500k_v2"
EXPECTED_TAGS = tuple(f"d2p{power:02d}" for power in range(6, 16))
EXPECTED_UPDATES = (300_000, 340_000, 380_000, 420_000, 460_000, 500_000)
DEFAULT_PREFIX = SWEEP
SUMMARY_NAME = "nf_generalize_fig2_dit_l16_continue500k_v2_physics_summary.csv"
SELECTED_NAME = "nf_generalize_fig2_dit_l16_continue500k_v2_pk_selected_bins.csv"
PATCH_NAME = "nf_generalize_fig2_dit_l16_continue500k_v2_patch_boundaries.csv"
CURVES_NAME = "nf_generalize_fig2_dit_l16_continue500k_v2_curves.npz"
SELECTED_K_BINS = (20, 40, 60)


def _project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"manifest must contain a list: {path}")
    return rows


def _data_signature(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError(f"config has no data mapping: {config_path}")
    return json.dumps(config["data"], sort_keys=True, default=str)


def _sample_label(updates: int) -> str:
    return "dpm50_source_300k" if updates == 300_000 else f"dpm50_cont_{updates // 1000}k"


def _selected_analysis_rows(
    rows: list[dict[str, Any]], labels: set[str] | None
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        updates = int(row.get("analysis_updates", row.get("target_total_updates", -1)))
        label = _sample_label(updates) if updates in EXPECTED_UPDATES else ""
        if label and (labels is None or label in labels):
            item = dict(row)
            item["analysis_updates"] = updates
            item["analysis_sample_label"] = label
            selected.append(item)

    expected_labels = set(labels or map(_sample_label, EXPECTED_UPDATES))
    expected_count = len(EXPECTED_TAGS) * len(expected_labels)
    if len(selected) != expected_count:
        raise ValueError(
            f"analysis manifest must provide {expected_count} selected rows; "
            f"found {len(selected)}"
        )
    pairs = {(str(row["dataset_tag"]), row["analysis_sample_label"]) for row in selected}
    expected_pairs = {(tag, label) for tag in EXPECTED_TAGS for label in expected_labels}
    if pairs != expected_pairs:
        missing = sorted(expected_pairs - pairs)
        extra = sorted(pairs - expected_pairs)
        raise ValueError(f"analysis manifest pair mismatch; missing={missing}, extra={extra}")
    return sorted(selected, key=lambda row: (int(row["dataset_size"]), int(row["analysis_updates"])))


def _continuation_run_names(rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        if int(row["analysis_updates"]) > 300_000:
            result[str(row["dataset_tag"])] = str(row["run_name"])
    if set(result) != set(EXPECTED_TAGS):
        raise ValueError("could not recover one continuation run name per dataset tag")
    return result


def _load_samples(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing sample artifact: {path}")
    with np.load(path) as data:
        if "samples" not in data.files:
            raise ValueError(f"sample artifact has no samples tensor: {path}")
        samples = np.asarray(data["samples"], dtype=np.float32).copy()
    if samples.shape != (512, 1, 128, 128):
        raise ValueError(f"unexpected sample shape {samples.shape}: {path}")
    if not np.isfinite(samples).all():
        raise ValueError(f"sample artifact contains non-finite values: {path}")
    return samples


def _concat_patch_series(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("no patch-boundary batches were accumulated")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _stream_reference(
    config_path: Path,
    *,
    hist_edges: np.ndarray,
    pk_nbins: int,
    k_max: float,
    raw_batch_size: int,
) -> dict[str, Any]:
    histogram = np.zeros(len(hist_edges) - 1, dtype=np.int64)
    pk_sum = np.zeros(pk_nbins, dtype=np.float64)
    pk_count = np.zeros(pk_nbins, dtype=np.int64)
    kbins: np.ndarray | None = None
    patch_parts: list[dict[str, np.ndarray]] = []
    selected_pk_parts: list[np.ndarray] = []
    n_images = 0
    total_pixel_count = 0
    split_histogram = np.zeros((2, len(hist_edges) - 1), dtype=np.int64)
    split_pk_sum = np.zeros((2, pk_nbins), dtype=np.float64)
    split_pk_count = np.zeros((2, pk_nbins), dtype=np.int64)
    split_image_count = np.zeros(2, dtype=np.int64)
    for batch in iter_real_reference_batches_from_config(
        config_path, raw_batch_size=raw_batch_size
    ):
        array = np.asarray(batch, dtype=np.float32)
        if array.ndim != 4 or array.shape[1:] != (1, 128, 128):
            raise ValueError(f"unexpected real-reference batch shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"real-reference batch contains non-finite values: {config_path}")
        histogram += np.histogram(array, bins=hist_edges)[0]
        total_pixel_count += int(array.size)
        spectra, current_kbins = batch_power_spectra(
            array, nbins=pk_nbins, k_max=k_max
        )
        if kbins is None:
            kbins = current_kbins
        elif not np.allclose(kbins, current_kbins, equal_nan=True):
            raise ValueError("real-reference k-bin definitions changed between batches")
        finite = np.isfinite(spectra)
        pk_sum += np.where(finite, spectra, 0.0).sum(axis=0)
        pk_count += finite.sum(axis=0)
        global_indices = n_images + np.arange(len(array))
        for split_index in (0, 1):
            split_mask = (global_indices % 2) == split_index
            if not np.any(split_mask):
                continue
            split_histogram[split_index] += np.histogram(
                array[split_mask], bins=hist_edges
            )[0]
            split_finite = finite[split_mask]
            split_spectra = spectra[split_mask]
            split_pk_sum[split_index] += np.where(
                split_finite, split_spectra, 0.0
            ).sum(axis=0)
            split_pk_count[split_index] += split_finite.sum(axis=0)
            split_image_count[split_index] += int(split_mask.sum())
        if max(SELECTED_K_BINS) >= spectra.shape[1]:
            raise ValueError("power spectrum has too few bins for selected-k analysis")
        selected_pk_parts.append(spectra[:, SELECTED_K_BINS])
        patch_parts.append(patch_boundary_per_image(array, patch_size=8))
        n_images += len(array)
    if n_images < 1 or kbins is None or np.any(pk_count == 0):
        raise ValueError(f"incomplete real-reference stream for {config_path}")
    in_range_count = int(histogram.sum())
    if in_range_count < 1 or total_pixel_count < 1:
        raise ValueError(f"reference histogram has no in-range pixels: {config_path}")
    probability = histogram / in_range_count
    patch_series = _concat_patch_series(patch_parts)
    real_floor = split_reference_physics_floor(
        histogram_a=split_histogram[0],
        histogram_b=split_histogram[1],
        pk_sum_a=split_pk_sum[0],
        pk_count_a=split_pk_count[0],
        pk_sum_b=split_pk_sum[1],
        pk_count_b=split_pk_count[1],
        n_images_a=int(split_image_count[0]),
        n_images_b=int(split_image_count[1]),
    )
    return {
        "n_images": n_images,
        "histogram_probability": probability,
        "pixel_coverage": float(in_range_count / total_pixel_count),
        "pk_mean": pk_sum / pk_count,
        "selected_pk": np.concatenate(selected_pk_parts, axis=0),
        "kbins": kbins,
        "patch_series": patch_series,
        "patch_summary": summarize_patch_boundary_series(patch_series, patch_size=8),
        "data_signature": _data_signature(config_path),
        **real_floor,
    }


def _power_summary(real_mean: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    ratio = np.nanmean(generated, axis=0) / np.clip(real_mean, 1.0e-30, None)
    finite = np.flatnonzero(np.isfinite(ratio))
    if not len(finite):
        raise ValueError("power-spectrum ratio has no finite bins")
    thirds = np.array_split(finite, 3)
    return {
        "pk_log10_mae": float(np.mean(np.abs(np.log10(np.clip(ratio[finite], 1.0e-30, None))))),
        "pk_ratio_low_k": float(np.mean(ratio[thirds[0]])),
        "pk_ratio_mid_k": float(np.mean(ratio[thirds[1]])),
        "pk_ratio_high_k": float(np.mean(ratio[thirds[2]])),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_curves(path: Path, curves: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **curves)
    os.replace(temporary, path)


def _baseline_patch_rows(
    project_dir: Path,
    manifest_path: Path,
    references: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _load_rows(manifest_path)
    selected = [row for row in rows if row.get("arch") in {"dit_l8", "dit_base"}]
    pairs = {(str(row.get("arch")), str(row.get("dataset_tag"))) for row in selected}
    expected = {(arch, tag) for arch in ("dit_l8", "dit_base") for tag in EXPECTED_TAGS}
    if len(selected) != len(expected) or pairs != expected:
        raise ValueError(
            "baseline manifest must contain exactly one 200k DiT-L8 and DiT-L12/base "
            "row for every dataset tag; "
            f"rows={len(selected)}, missing={sorted(expected - pairs)}, "
            f"extra={sorted(pairs - expected)}"
        )
    output = []
    labels = {"dit_l8": "DiT-L8", "dit_base": "DiT-L12 / base"}
    for row in sorted(selected, key=lambda item: (str(item["arch"]), int(item["dataset_size"]))):
        tag = str(row["dataset_tag"])
        config_path = _project_path(project_dir, row["config"])
        if _data_signature(config_path) != references[tag]["data_signature"]:
            raise ValueError(f"{row['arch']} {tag} does not use the same exact training subset")
        sample_value = str(row["sample_path"]).format(seed=123, sample_label="dpm50")
        sample_path = _project_path(project_dir, sample_value)
        samples = _load_samples(sample_path)
        stats = patch_boundary_statistics(samples, patch_size=8)
        output.append(
            {
                "architecture": row["arch"],
                "architecture_label": labels[str(row["arch"])],
                "dataset_tag": tag,
                "dataset_size": int(row["dataset_size"]),
                "updates_k": 200,
                "sample_label": "dpm50",
                "sample_path": str(sample_path),
                "reference_kind": "generated",
                **stats,
            }
        )
    return output


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    project_dir = Path(args.project_dir).resolve()
    manifest_path = _project_path(project_dir, args.manifest)
    baseline_manifest = _project_path(project_dir, args.baseline_manifest)
    sample_root = _project_path(project_dir, args.sample_root)
    table_dir = _project_path(project_dir, args.table_dir)
    physics_dir = _project_path(project_dir, args.physics_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    physics_dir.mkdir(parents=True, exist_ok=True)
    labels = set(args.sample_label or ()) or None
    all_analysis_rows = _load_rows(manifest_path)
    analysis_rows = _selected_analysis_rows(all_analysis_rows, labels)
    run_names = _continuation_run_names(
        _selected_analysis_rows(all_analysis_rows, labels=None)
    )
    if int(args.hist_bins) == 140:
        hist_edges = PHYSICAL_HIST_EDGES.copy()
        if not np.array_equal(
            hist_edges,
            np.linspace(-1.0, 1.0, int(args.hist_bins) + 1, dtype=np.float64),
        ):
            raise AssertionError("default histogram edges differ from PHYSICAL_HIST_EDGES")
    else:
        hist_edges = np.linspace(
            -1.0, 1.0, int(args.hist_bins) + 1, dtype=np.float64
        )

    references: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    curves: dict[str, np.ndarray] = {"histogram_edges": hist_edges}

    for row in analysis_rows:
        tag = str(row["dataset_tag"])
        updates = int(row["analysis_updates"])
        label = str(row["analysis_sample_label"])
        config_value = (
            row.get("source_config") or row.get("config")
            if updates == 300_000
            else row.get("config")
        )
        if not config_value:
            raise ValueError(f"{tag} {label}: no config path in analysis manifest")
        config_path = _project_path(project_dir, config_value)
        if tag not in references:
            references[tag] = _stream_reference(
                config_path,
                hist_edges=hist_edges,
                pk_nbins=int(args.pk_nbins),
                k_max=float(args.k_max),
                raw_batch_size=int(args.raw_batch_size),
            )
            reference_patch = dict(references[tag]["patch_summary"])
            patch_rows.append(
                {
                    "architecture": "real_reference",
                    "architecture_label": "exact training subset",
                    "dataset_tag": tag,
                    "dataset_size": int(row["dataset_size"]),
                    "updates_k": "",
                    "sample_label": "",
                    "sample_path": "",
                    "reference_kind": "real",
                    **reference_patch,
                }
            )
        reference = references[tag]
        if _data_signature(config_path) != reference["data_signature"]:
            raise ValueError(f"{tag} {label} changed the configured training subset")

        sample_path = sample_root / f"{run_names[tag]}_seed123_{label}.npz"
        checkpoint = Path(row.get("analysis_checkpoint", row.get("expected_checkpoint")))
        audit = validate_sample_file(
            sample_path,
            requested_checkpoint=checkpoint,
            scheduler="DPMSolverMultistepScheduler",
            requested_steps=50,
        )
        samples = _load_samples(sample_path)
        generated_probability, generated_pixel_coverage = histogram_probability_and_coverage(
            samples, hist_edges
        )
        hist_l1 = float(
            np.sum(np.abs(reference["histogram_probability"] - generated_probability))
        )
        hist_interval = bootstrap_histogram_l1_interval(
            samples,
            reference["histogram_probability"],
            hist_edges,
            n_resamples=int(args.bootstrap_resamples),
            seed=int(args.seed),
        )
        generated_pk, kbins = batch_power_spectra(
            samples, nbins=int(args.pk_nbins), k_max=float(args.k_max)
        )
        if not np.allclose(kbins, reference["kbins"], equal_nan=True):
            raise ValueError(f"{tag} {label}: generated and real k-bins differ")
        power = _power_summary(reference["pk_mean"], generated_pk)
        power_interval = bootstrap_power_log10_mae_interval(
            generated_pk,
            reference["pk_mean"],
            n_resamples=int(args.bootstrap_resamples),
            seed=int(args.seed),
        )
        patch = patch_boundary_statistics(samples, patch_size=8)
        patch_rows.append(
            {
                "architecture": "dit_l16",
                "architecture_label": "DiT-L16",
                "dataset_tag": tag,
                "dataset_size": int(row["dataset_size"]),
                "updates_k": updates // 1000,
                "sample_label": label,
                "sample_path": str(sample_path),
                "reference_kind": "generated",
                **patch,
            }
        )
        summary_rows.append(
            {
                "sweep_name": SWEEP,
                "architecture": "dit_l16",
                "dataset_tag": tag,
                "dataset_size": int(row["dataset_size"]),
                "updates_k": updates // 1000,
                "sample_label": label,
                "sample_path": str(sample_path),
                "config_path": str(config_path),
                "checkpoint": str(checkpoint),
                "n_generated": len(samples),
                "n_real_exact_subset": int(reference["n_images"]),
                "k_max": float(args.k_max),
                "hist_bins": int(args.hist_bins),
                "hist_min": float(hist_edges[0]),
                "hist_max": float(hist_edges[-1]),
                "real_pixel_coverage": float(reference["pixel_coverage"]),
                "generated_pixel_coverage": generated_pixel_coverage,
                "hist_l1": hist_l1,
                "hist_l1_lo": hist_interval[0],
                "hist_l1_hi": hist_interval[1],
                "pk_log10_mae_lo": power_interval[0],
                "pk_log10_mae_hi": power_interval[1],
                "bootstrap_resamples": int(args.bootstrap_resamples),
                "bootstrap_seed": int(args.seed),
                "real_vs_real_hist_l1": reference["real_vs_real_hist_l1"],
                "real_vs_real_pk_log10_mae": reference["real_vs_real_pk_log10_mae"],
                "n_real_half_a": reference["real_half_a_count"],
                "n_real_half_b": reference["real_half_b_count"],
                **power,
                "patch_boundary_ratio": patch["boundary_to_control_ratio"],
                "scheduler": audit["scheduler"],
                "executed_inference_steps": audit["executed_inference_steps"],
                "terminal_sigma": audit["terminal_sigma"],
                "terminal_sigma_verifiable": audit["terminal_sigma_verifiable"],
            }
        )

        ratio_samples = generated_pk / np.clip(reference["pk_mean"][None, :], 1.0e-30, None)
        raw_stats = selected_power_bin_statistics(
            generated_pk,
            bin_indices=SELECTED_K_BINS,
            n_resamples=int(args.bootstrap_resamples),
            seed=int(args.seed),
        )
        ratio_stats = two_sample_selected_power_ratio_statistics(
            generated_pk,
            reference["selected_pk"],
            bin_indices=SELECTED_K_BINS,
            n_resamples=int(args.bootstrap_resamples),
            seed=int(args.seed),
        )
        for raw, ratio in zip(raw_stats, ratio_stats):
            index = int(raw["k_bin"])
            selected_rows.append(
                {
                    "dataset_tag": tag,
                    "dataset_size": int(row["dataset_size"]),
                    "updates_k": updates // 1000,
                    "sample_label": label,
                    "k_bin": index,
                    "k_value": float(kbins[index]),
                    "real_reference_mean": float(reference["pk_mean"][index]),
                    "real_pk_sem": ratio["real_pk_sem"],
                    **{f"generated_{key}": value for key, value in raw.items() if key != "k_bin"},
                    **{
                        f"ratio_{key}": value
                        for key, value in ratio.items()
                        if key not in {"k_bin", "real_pk_sem"}
                    },
                }
            )

        key = f"{tag}_{updates // 1000}k"
        curves[f"{key}_kbins"] = kbins
        curves[f"{key}_real_hist_probability"] = reference["histogram_probability"]
        curves[f"{key}_generated_hist_probability"] = generated_probability
        curves[f"{key}_real_pk_mean"] = reference["pk_mean"]
        curves[f"{key}_generated_pk_mean"] = np.nanmean(generated_pk, axis=0)
        curves[f"{key}_pk_ratio"] = np.nanmean(ratio_samples, axis=0)

    patch_rows.extend(_baseline_patch_rows(project_dir, baseline_manifest, references))

    prefix = str(args.out_prefix)
    if prefix == DEFAULT_PREFIX:
        paths = {
            "summary": table_dir / SUMMARY_NAME,
            "selected": table_dir / SELECTED_NAME,
            "patch": table_dir / PATCH_NAME,
            "curves": physics_dir / CURVES_NAME,
        }
    else:
        paths = {
            "summary": table_dir / f"{prefix}_physics_summary.csv",
            "selected": table_dir / f"{prefix}_pk_selected_bins.csv",
            "patch": table_dir / f"{prefix}_patch_boundaries.csv",
            "curves": physics_dir / f"{prefix}_curves.npz",
        }
    _write_csv(paths["summary"], summary_rows)
    _write_csv(paths["selected"], selected_rows)
    _write_csv(paths["patch"], patch_rows)
    _write_curves(paths["curves"], curves)
    for name, path in paths.items():
        print(f"wrote {name}: {path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument(
        "--manifest",
        default=f"local/{SWEEP}/analysis_manifest.json",
    )
    parser.add_argument(
        "--baseline-manifest", default="local/nf_generalize_fig2_dit/manifest.json"
    )
    parser.add_argument("--sample-root", default=f"results/{SWEEP}/samples")
    parser.add_argument("--table-dir", default="results/nf_generalize_fig2_dit/tables")
    parser.add_argument("--physics-dir", default="results/nf_generalize_fig2_dit/physics")
    parser.add_argument("--sample-label", action="append")
    parser.add_argument("--out-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--raw-batch-size", type=int, default=4)
    parser.add_argument("--pk-nbins", type=int, default=91)
    parser.add_argument("--k-max", type=float, default=64.0)
    parser.add_argument("--hist-bins", type=int, default=140)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
