#!/usr/bin/env python
"""Fail closed unless one fresh DiT-L16 milestone is complete and provenance-safe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_manifest(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return rows


def scalar_text(archive: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in archive:
        raise ValueError(f"Sample archive is missing provenance field {key!r}")
    return str(np.asarray(archive[key]).item())


def audit_metrics(path: Path, expected_tags: set[str]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing analysis table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    row_tags = [row.get("dataset_tag", "") for row in rows]
    duplicate_tags = sorted({tag for tag in row_tags if row_tags.count(tag) > 1})
    if duplicate_tags:
        raise ValueError(f"{path.name} has duplicate dataset tags: {duplicate_tags}")
    tags = set(row_tags)
    if tags != expected_tags:
        raise ValueError(
            f"{path.name} has dataset tags {sorted(tags)}, expected {sorted(expected_tags)}"
        )
    if len(rows) != len(expected_tags):
        raise ValueError(
            f"{path.name} has {len(rows)} rows, expected {len(expected_tags)}"
        )
    return {
        "path": str(path),
        "row_count": len(rows),
        "dataset_tags": sorted(tags),
    }


def audit(args: argparse.Namespace) -> dict:
    project_dir = args.project_dir.resolve()
    all_rows = read_manifest(args.manifest.resolve())
    rows = [
        row
        for row in all_rows
        if int(row["target_total_updates"]) == int(args.updates)
    ]
    if len(rows) != 10:
        raise ValueError(
            f"Expected ten manifest rows at {args.updates} updates, found {len(rows)}"
        )
    if not all(row["scientific_checkpoint"] for row in rows):
        raise ValueError(f"{args.updates} is not a frozen scientific milestone")

    expected_tags = {row["dataset_tag"] for row in rows}
    sample_label = f"dpm50_fresh_{args.updates // 1000}k"
    sample_root = project_dir / "results" / "nf_generalize_fig2_dit_l16_fresh300k" / "samples"
    expected_sample_shape = (512, 1, 128, 128)
    sample_summaries = []
    for row in rows:
        checkpoint = Path(row["expected_checkpoint"]).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing exact checkpoint: {checkpoint}")
        sample_path = sample_root / (
            f"{row['run_name']}_seed123_{sample_label}.npz"
        )
        if not sample_path.is_file():
            raise FileNotFoundError(f"Missing exact-checkpoint sample: {sample_path}")
        with np.load(sample_path, allow_pickle=False) as archive:
            samples = np.asarray(archive["samples"])
            if samples.shape != expected_sample_shape:
                raise ValueError(
                    f"{sample_path} has sample shape {samples.shape}, "
                    f"expected {expected_sample_shape}"
                )
            if not np.isfinite(samples).all():
                raise ValueError(f"{sample_path} contains non-finite sample values")
            requested = Path(scalar_text(archive, "requested_checkpoint")).resolve()
            resolved = Path(scalar_text(archive, "resolved_checkpoint")).resolve()
            if requested != checkpoint or resolved != checkpoint:
                raise ValueError(
                    f"{sample_path} provenance does not point to exact checkpoint {checkpoint}"
                )
            if int(np.asarray(archive["num_steps"]).item()) != 50:
                raise ValueError(f"{sample_path} was not generated with 50 sampler steps")
            if int(np.asarray(archive["seed"]).item()) != 123:
                raise ValueError(f"{sample_path} was not generated with seed 123")
            scheduler = scalar_text(archive, "scheduler")
            if scheduler != "DPMSolverMultistepScheduler":
                raise ValueError(
                    f"{sample_path} uses scheduler {scheduler!r}, "
                    "expected 'DPMSolverMultistepScheduler'"
                )
            sample_summaries.append(
                {
                    "dataset_tag": row["dataset_tag"],
                    "sample_path": str(sample_path),
                    "checkpoint": str(checkpoint),
                    "shape": list(samples.shape),
                    "scheduler": scheduler,
                    "num_steps": 50,
                    "seed": 123,
                }
            )

    table_dir = project_dir / "results" / "nf_generalize_fig2_dit" / "tables"
    prefix = f"nf_generalize_fig2_dit_l16_fresh300k_{args.updates // 1000}k"
    pca_summary = audit_metrics(
        table_dir / f"{prefix}_pca_full_nn_metrics.csv", expected_tags
    )
    sscd_summary = audit_metrics(
        table_dir / f"{prefix}_sscd_full_nn_metrics.csv", expected_tags
    )
    summary = {
        "status": "pass",
        "updates": int(args.updates),
        "sample_label": sample_label,
        "expected_dataset_tags": sorted(expected_tags),
        "checkpoint_count": len(rows),
        "sample_count": len(sample_summaries),
        "samples": sample_summaries,
        "pca": pca_summary,
        "sscd": sscd_summary,
    }
    summary_output = args.summary_output
    if summary_output is None:
        summary_output = (
            project_dir
            / "results"
            / "nf_generalize_fig2_dit_l16_fresh300k"
            / "audits"
            / f"fresh_l16_{args.updates // 1000}k_audit.json"
        )
    summary_output = summary_output.resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Fresh DiT-L16 {args.updates // 1000}k audit PASS: "
        "10 checkpoints, 10 sample archives, PCA, and SSCD. "
        f"Summary: {summary_output}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--updates",
        required=True,
        type=int,
        choices=(200_000, 225_000, 250_000, 275_000, 300_000),
    )
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
