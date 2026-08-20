#!/usr/bin/env python
"""Audit DiT-L16 analysis readiness separately from checkpoint retention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_IMPORT_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, PROJECT_IMPORT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import check_nf_generalize_fig2_dit_l16_continue500k_v2 as precheck
import prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs as prep
from dit_300k_scaling_analysis import resolve_checkpoint_loss_metrics
from validate_nf_generalize_fig2_dit_sample import validate_sample_file


SWEEP = prep.CONTINUE_SWEEP_NAME
EXPECTED_UPDATES = (300_000,) + prep.TARGET_UPDATES
EXPECTED_TAGS = prep.EXPECTED_TAGS
EXPECTED_SAMPLE_SHAPE = (512, 1, 128, 128)
DPM_LABELS = {
    300_000: "dpm50_source_300k",
    **{updates: f"dpm50_cont_{updates // 1000}k" for updates in prep.TARGET_UPDATES},
}
DDPM_CONTROLS = (
    ("d2p08", 300_000, "ddpm500_source_300k"),
    ("d2p08", 500_000, "ddpm500_cont_500k"),
    ("d2p11", 300_000, "ddpm500_source_300k"),
    ("d2p11", 500_000, "ddpm500_cont_500k"),
)


def _project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"manifest must contain a list: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sample_digest(path: Path) -> tuple[str, str]:
    with np.load(path) as data:
        samples = np.ascontiguousarray(data["samples"])
        checkpoint = str(data["resolved_checkpoint"].item())
    return hashlib.sha256(samples.view(np.uint8)).hexdigest(), checkpoint


def _analysis_rows_by_pair(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    for row in rows:
        updates = int(row.get("analysis_updates", row.get("target_total_updates", -1)))
        pair = (str(row.get("dataset_tag", "")), updates)
        if pair in result:
            duplicates.append(pair)
        result[pair] = row
    expected = {(tag, updates) for tag in EXPECTED_TAGS for updates in EXPECTED_UPDATES}
    if duplicates or set(result) != expected or len(rows) != len(expected):
        raise ValueError(
            "analysis manifest must contain exactly one row for every dataset/update pair; "
            f"duplicates={sorted(set(duplicates))}, "
            f"missing={sorted(expected - set(result))}, extra={sorted(set(result) - expected)}"
        )
    return result


def _continuation_rows_by_pair(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    for row in rows:
        pair = (str(row.get("dataset_tag", "")), int(row.get("target_total_updates", -1)))
        if pair in result:
            duplicates.append(pair)
        result[pair] = row
    expected = {(tag, updates) for tag in EXPECTED_TAGS for updates in prep.TARGET_UPDATES}
    if duplicates or set(result) != expected or len(rows) != len(expected):
        raise ValueError(
            "continuation manifest must contain exactly one row for every target; "
            f"duplicates={sorted(set(duplicates))}, "
            f"missing={sorted(expected - set(result))}, extra={sorted(set(result) - expected)}"
        )
    return result


def _audit_checkpoint(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.is_dir():
        return [f"missing checkpoint directory: {path}"]
    for group, alternatives in precheck.REQUIRED_STATE_GROUPS.items():
        if precheck._state_path(path, alternatives) is None:
            issues.append(
                f"checkpoint {path} lacks {group}; expected one of {', '.join(alternatives)}"
            )
    return issues


def _audit_metric_table(path: Path, *, sample_label: str) -> list[str]:
    if not path.is_file():
        return [f"missing metric table: {path}"]
    rows = _read_csv(path)
    tags = [row.get("dataset_tag", "") for row in rows]
    issues: list[str] = []
    if len(rows) != len(EXPECTED_TAGS) or sorted(tags) != sorted(EXPECTED_TAGS):
        issues.append(
            f"metric table {path} must contain ten unique dataset tags; found {tags}"
        )
    for row in rows:
        if sample_label not in row.get("sample_path", ""):
            issues.append(
                f"metric table {path} row {row.get('dataset_tag')} does not point to {sample_label}"
            )
    return issues


def _audit_physics_outputs(project_dir: Path) -> tuple[dict[str, int], list[str]]:
    table_dir = project_dir / "results" / "nf_generalize_fig2_dit" / "tables"
    physics_dir = project_dir / "results" / "nf_generalize_fig2_dit" / "physics"
    summary_path = table_dir / f"{SWEEP}_physics_summary.csv"
    selected_path = table_dir / f"{SWEEP}_pk_selected_bins.csv"
    patch_path = table_dir / f"{SWEEP}_patch_boundaries.csv"
    curves_path = physics_dir / f"{SWEEP}_curves.npz"
    counts: dict[str, int] = {}
    issues: list[str] = []

    expected_pairs = {(tag, updates // 1000) for tag in EXPECTED_TAGS for updates in EXPECTED_UPDATES}
    if not summary_path.is_file():
        issues.append(f"missing physics summary: {summary_path}")
    else:
        rows = _read_csv(summary_path)
        pairs = {(row.get("dataset_tag", ""), int(row.get("updates_k", -1))) for row in rows}
        counts["physics_summary_rows"] = len(rows)
        if len(rows) != 60 or pairs != expected_pairs:
            issues.append("physics summary does not contain all 60 dataset/checkpoint rows")
        low_coverage: list[tuple[str, int]] = []
        invalid_coverage: list[tuple[str, int]] = []
        for row in rows:
            pair = (row.get("dataset_tag", ""), int(row.get("updates_k", -1)))
            try:
                coverage = float(row["generated_pixel_coverage"])
            except (KeyError, TypeError, ValueError):
                invalid_coverage.append(pair)
                continue
            if not np.isfinite(coverage):
                invalid_coverage.append(pair)
            elif coverage < 0.999:
                low_coverage.append(pair)
        if invalid_coverage:
            issues.append(
                "missing or invalid generated pixel coverage: "
                f"{sorted(invalid_coverage)}"
            )
        if low_coverage:
            issues.append(
                "generated pixel coverage below 0.999: "
                f"{sorted(low_coverage)}"
            )

    if not selected_path.is_file():
        issues.append(f"missing selected-bin table: {selected_path}")
    else:
        rows = _read_csv(selected_path)
        triples = {
            (row.get("dataset_tag", ""), int(row.get("updates_k", -1)), int(row.get("k_bin", -1)))
            for row in rows
        }
        expected_triples = {
            (tag, updates // 1000, kbin)
            for tag in EXPECTED_TAGS
            for updates in EXPECTED_UPDATES
            for kbin in (20, 40, 60)
        }
        counts["selected_bin_rows"] = len(rows)
        if len(rows) != 180 or triples != expected_triples:
            issues.append("selected-bin table does not contain all 180 required rows")

    if not patch_path.is_file():
        issues.append(f"missing patch-boundary table: {patch_path}")
    else:
        rows = _read_csv(patch_path)
        counts["patch_boundary_rows"] = len(rows)
        l16_pairs = {
            (row.get("dataset_tag", ""), int(row.get("updates_k", -1)))
            for row in rows
            if row.get("architecture") == "dit_l16"
        }
        real_tags = {
            row.get("dataset_tag", "")
            for row in rows
            if row.get("architecture") == "real_reference"
        }
        baseline_pairs = {
            (row.get("architecture", ""), row.get("dataset_tag", ""))
            for row in rows
            if row.get("architecture") in {"dit_l8", "dit_base"}
        }
        expected_baselines = {
            (arch, tag) for arch in ("dit_l8", "dit_base") for tag in EXPECTED_TAGS
        }
        if l16_pairs != expected_pairs or real_tags != set(EXPECTED_TAGS) or baseline_pairs != expected_baselines:
            issues.append("patch-boundary table lacks required L16, real, or L8/L12 rows")

    if not curves_path.is_file():
        issues.append(f"missing physics curves: {curves_path}")
    else:
        with np.load(curves_path) as curves:
            keys = set(curves.files)
        required = {
            f"{tag}_{updates // 1000}k_{suffix}"
            for tag in EXPECTED_TAGS
            for updates in EXPECTED_UPDATES
            for suffix in (
                "kbins",
                "real_hist_probability",
                "generated_hist_probability",
                "real_pk_mean",
                "generated_pk_mean",
                "pk_ratio",
            )
        }
        counts["physics_curve_arrays"] = len(keys)
        if not required.issubset(keys):
            issues.append(f"physics curves are missing {len(required - keys)} required arrays")

    return counts, issues


def audit_results(project_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Audit the complete sweep and always write a machine-readable report."""
    project_dir = Path(project_dir).resolve()
    manifest_path = _project_path(project_dir, manifest_path)
    report_path = project_dir / "local" / SWEEP / "final_audit.json"
    report: dict[str, Any] = {
        "sweep_name": SWEEP,
        "status": "FAIL",
        "analysis_status": "FAIL",
        "checkpoint_retention_status": "INCOMPLETE",
        "counts": {},
        "missing_paths": [],
        "retention_missing_paths": [],
        "provenance_mismatches": [],
        "duplicate_hashes": [],
        "issues": [],
        "retention_issues": [],
        "warnings": [],
        "loss_metric_sources": [],
    }

    try:
        continuation_rows = _load_json_rows(manifest_path)
        continuation = _continuation_rows_by_pair(continuation_rows)
        analysis_path = manifest_path.with_name("analysis_manifest.json")
        analysis_rows = _load_json_rows(analysis_path)
        analysis = _analysis_rows_by_pair(analysis_rows)
    except Exception as exc:
        report["issues"].append(str(exc))
        _atomic_json(report_path, report)
        return report

    final_target = max(prep.TARGET_UPDATES)
    valid_checkpoints = 0
    valid_final_checkpoints = 0
    valid_intermediate_checkpoints = 0
    for (_, updates), row in sorted(continuation.items()):
        issues = _audit_checkpoint(Path(row["expected_checkpoint"]))
        if not issues:
            valid_checkpoints += 1
            if updates == final_target:
                valid_final_checkpoints += 1
            else:
                valid_intermediate_checkpoints += 1
            continue

        report["retention_issues"].extend(issues)
        missing = [
            issue.removeprefix("missing checkpoint directory: ")
            for issue in issues
            if issue.startswith("missing checkpoint directory")
        ]
        report["retention_missing_paths"].extend(missing)
        if updates == final_target:
            report["issues"].extend(issues)
            report["missing_paths"].extend(missing)

    report["counts"]["expected_checkpoints"] = 50
    report["counts"]["valid_checkpoints"] = valid_checkpoints
    report["counts"]["expected_final_checkpoints"] = len(EXPECTED_TAGS)
    report["counts"]["valid_final_checkpoints"] = valid_final_checkpoints
    report["counts"]["expected_intermediate_checkpoints"] = (
        len(EXPECTED_TAGS) * (len(prep.TARGET_UPDATES) - 1)
    )
    report["counts"]["valid_intermediate_checkpoints"] = (
        valid_intermediate_checkpoints
    )
    report["counts"]["missing_intermediate_checkpoints"] = (
        report["counts"]["expected_intermediate_checkpoints"]
        - valid_intermediate_checkpoints
    )
    if report["retention_issues"]:
        report["warnings"].append(
            "Checkpoint retention is incomplete. Derived artifacts remain analyzable "
            "only when analysis_status is PASS; missing snapshots cannot be regenerated "
            "or resumed from their exact weights."
        )
    else:
        report["checkpoint_retention_status"] = "PASS"

    durable_metrics_dir = project_dir / "results" / SWEEP / "training_metrics"
    log_dir = project_dir / "logs" / SWEEP
    valid_loss_metric_sources = 0
    for (tag, updates), row in sorted(continuation.items()):
        try:
            resolved = resolve_checkpoint_loss_metrics(
                row,
                durable_metrics_dir=durable_metrics_dir,
                log_dir=log_dir,
            )
        except (FileNotFoundError, ValueError) as exc:
            report["issues"].append(str(exc))
            continue
        valid_loss_metric_sources += 1
        report["loss_metric_sources"].append(
            {
                "dataset_tag": tag,
                "updates_k": updates // 1000,
                "source_kind": resolved.source_kind,
                "source_paths": [str(path) for path in resolved.source_paths],
                "first_epoch": resolved.first_epoch,
                "final_epoch": resolved.final_epoch,
            }
        )
    report["counts"]["expected_loss_metric_sources"] = len(continuation)
    report["counts"]["valid_loss_metric_sources"] = valid_loss_metric_sources

    sample_paths: list[Path] = []
    sample_hashes: dict[str, tuple[Path, str]] = {}
    for pair, row in sorted(analysis.items()):
        updates = pair[1]
        path = _project_path(project_dir, row["sample_path"])
        checkpoint = Path(row.get("analysis_checkpoint", row.get("expected_checkpoint")))
        try:
            validate_sample_file(
                path,
                requested_checkpoint=checkpoint,
                scheduler="DPMSolverMultistepScheduler",
                requested_steps=50,
                expected_shape=EXPECTED_SAMPLE_SHAPE,
            )
            digest, resolved = _sample_digest(path)
            previous = sample_hashes.get(digest)
            if previous is not None and previous[1] != resolved:
                report["duplicate_hashes"].append(
                    {
                        "sha256": digest,
                        "first": str(previous[0]),
                        "second": str(path),
                        "first_checkpoint": previous[1],
                        "second_checkpoint": resolved,
                    }
                )
            else:
                sample_hashes[digest] = (path, resolved)
            sample_paths.append(path)
        except FileNotFoundError:
            report["missing_paths"].append(str(path))
        except Exception as exc:
            report["provenance_mismatches"].append(f"{path}: {exc}")

    run_names = {
        tag: str(continuation[(tag, prep.TARGET_UPDATES[0])]["run_name"])
        for tag in EXPECTED_TAGS
    }
    sample_root = project_dir / "results" / SWEEP / "samples"
    for tag, updates, label in DDPM_CONTROLS:
        path = sample_root / f"{run_names[tag]}_seed123_{label}.npz"
        checkpoint = Path(analysis[(tag, updates)]["analysis_checkpoint"])
        try:
            validate_sample_file(
                path,
                requested_checkpoint=checkpoint,
                scheduler="DDPMScheduler",
                requested_steps=500,
                expected_shape=EXPECTED_SAMPLE_SHAPE,
            )
            digest, resolved = _sample_digest(path)
            previous = sample_hashes.get(digest)
            if previous is not None and previous[1] != resolved:
                report["duplicate_hashes"].append(
                    {
                        "sha256": digest,
                        "first": str(previous[0]),
                        "second": str(path),
                        "first_checkpoint": previous[1],
                        "second_checkpoint": resolved,
                    }
                )
            else:
                sample_hashes[digest] = (path, resolved)
            sample_paths.append(path)
        except FileNotFoundError:
            report["missing_paths"].append(str(path))
        except Exception as exc:
            report["provenance_mismatches"].append(f"{path}: {exc}")
    report["counts"]["expected_dpm_samples"] = 60
    report["counts"]["expected_ddpm_controls"] = 4
    report["counts"]["valid_sample_files"] = len(sample_paths)

    table_dir = project_dir / "results" / "nf_generalize_fig2_dit" / "tables"
    metric_issues: list[str] = []
    valid_metric_tables = 0
    for updates in EXPECTED_UPDATES:
        label = DPM_LABELS[updates]
        for feature in ("pca", "sscd"):
            path = table_dir / f"{SWEEP}_{updates // 1000}k_{feature}_full_nn_metrics.csv"
            issues = _audit_metric_table(path, sample_label=label)
            metric_issues.extend(issues)
            if not issues:
                valid_metric_tables += 1
    report["counts"]["expected_metric_tables"] = 12
    report["counts"]["valid_metric_tables"] = valid_metric_tables
    report["missing_paths"].extend(
        issue.removeprefix("missing metric table: ")
        for issue in metric_issues
        if issue.startswith("missing metric table")
    )
    report["issues"].extend(metric_issues)

    physics_counts, physics_issues = _audit_physics_outputs(project_dir)
    report["counts"].update(physics_counts)
    report["missing_paths"].extend(
        issue.split(": ", 1)[1]
        for issue in physics_issues
        if issue.startswith("missing ") and ": " in issue
    )
    report["issues"].extend(physics_issues)

    report["missing_paths"] = sorted(set(report["missing_paths"]))
    report["retention_missing_paths"] = sorted(
        set(report["retention_missing_paths"])
    )
    report["provenance_mismatches"] = sorted(
        set(report["provenance_mismatches"])
    )
    report["issues"] = sorted(set(report["issues"]))
    report["retention_issues"] = sorted(set(report["retention_issues"]))
    report["warnings"] = sorted(set(report["warnings"]))

    if not (
        report["issues"]
        or report["missing_paths"]
        or report["provenance_mismatches"]
        or report["duplicate_hashes"]
    ):
        report["analysis_status"] = "PASS"
        report["status"] = (
            "PASS"
            if report["checkpoint_retention_status"] == "PASS"
            else "PASS_WITH_WARNINGS"
        )
    _atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument(
        "--manifest", default=f"local/{SWEEP}/manifest.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_results(Path(args.project_dir), Path(args.manifest))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["analysis_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
