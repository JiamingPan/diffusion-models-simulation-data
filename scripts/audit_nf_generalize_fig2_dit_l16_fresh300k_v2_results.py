#!/usr/bin/env python
"""Fail closed unless the clean DiT-L16 300k result set is complete."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SWEEP_NAME = "nf_generalize_fig2_dit_l16_fresh300k_v2"
SAMPLE_LABEL = "dpm50_fresh300k_v2"


def audit_metrics(path: Path, expected_tags: set[str]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing analysis table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    tags = [row.get("dataset_tag", "") for row in rows]
    if len(tags) != 10 or set(tags) != expected_tags or len(set(tags)) != len(tags):
        raise ValueError(f"{path.name} does not contain one row for every data size")
    return {"path": str(path), "row_count": len(rows)}


def scalar_text(archive: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in archive:
        raise ValueError(f"Sample archive is missing provenance field {key!r}")
    return str(np.asarray(archive[key]).item())


def audit(args: argparse.Namespace) -> dict:
    project_dir = args.project_dir.resolve()
    rows = json.loads(args.manifest.resolve().read_text())
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("Expected ten clean 300k manifest rows")
    expected_tags = {row["dataset_tag"] for row in rows}
    sample_root = project_dir / "results" / SWEEP_NAME / "samples"
    samples_summary = []

    for row in rows:
        checkpoint = Path(row["expected_checkpoint"]).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing exact checkpoint: {checkpoint}")
        sample_path = sample_root / (
            f"{row['run_name']}_seed123_{SAMPLE_LABEL}.npz"
        )
        if not sample_path.is_file():
            raise FileNotFoundError(f"Missing sample archive: {sample_path}")
        with np.load(sample_path, allow_pickle=False) as archive:
            samples = np.asarray(archive["samples"])
            if samples.shape != (512, 1, 128, 128):
                raise ValueError(f"Unexpected sample shape in {sample_path}: {samples.shape}")
            if not np.isfinite(samples).all():
                raise ValueError(f"Non-finite sample values in {sample_path}")
            requested = Path(scalar_text(archive, "requested_checkpoint")).resolve()
            resolved = Path(scalar_text(archive, "resolved_checkpoint")).resolve()
            if requested != checkpoint or resolved != checkpoint:
                raise ValueError(f"Sample provenance mismatch in {sample_path}")
            if int(np.asarray(archive["num_steps"]).item()) != 50:
                raise ValueError(f"Wrong sampler step count in {sample_path}")
            if int(np.asarray(archive["seed"]).item()) != 123:
                raise ValueError(f"Wrong sampling seed in {sample_path}")
        samples_summary.append(
            {
                "dataset_tag": row["dataset_tag"],
                "checkpoint": str(checkpoint),
                "sample_path": str(sample_path),
            }
        )

    table_dir = project_dir / "results" / "nf_generalize_fig2_dit" / "tables"
    prefix = SWEEP_NAME
    summary = {
        "status": "pass",
        "target_total_updates": 300_000,
        "checkpoint_count": 10,
        "sample_count": 10,
        "samples": samples_summary,
        "pca": audit_metrics(
            table_dir / f"{prefix}_pca_full_nn_metrics.csv", expected_tags
        ),
        "sscd": audit_metrics(
            table_dir / f"{prefix}_sscd_full_nn_metrics.csv", expected_tags
        ),
    }
    output = (
        project_dir
        / "results"
        / SWEEP_NAME
        / "audits"
        / "fresh_l16_300k_v2_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Fresh DiT-L16 300k v2 audit PASS: {output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
