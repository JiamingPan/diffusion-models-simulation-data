#!/usr/bin/env python
"""Audit the ten-size conditional calibration sweep before or after execution."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from prepare_nf_conditional_bias_full_sweep_configs import (
    ALL_DATASET_SIZES,
    CHECKPOINT_ROOT,
    SWEEP_NAME,
    TRAINING_SEED,
)
from prepare_nf_conditional_u128_config import PARAM_NAMES


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-epoch-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _latest_checkpoint(checkpoint_dir: Path) -> tuple[Path, int]:
    candidates = []
    for path in checkpoint_dir.glob("checkpoint-epoch-*"):
        epoch = _checkpoint_epoch(path)
        if path.is_dir() and epoch is not None:
            candidates.append((path, epoch))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-epoch-* directories in {checkpoint_dir}")
    return max(candidates, key=lambda item: item[1])


def _require_pinned_checkpoint(row: dict[str, Any]) -> Path:
    expected = int(row.get("checkpoint_epoch", int(row["epochs"]) - 1))
    raw_path = row.get("requested_checkpoint")
    path = (
        Path(str(raw_path))
        if raw_path is not None
        else Path(row["checkpoint_dir"]) / f"checkpoint-epoch-{expected:04d}"
    )
    if not path.is_dir():
        raise FileNotFoundError(f"{row['run_name']} missing pinned checkpoint {path}")
    epoch = _checkpoint_epoch(path)
    if epoch != expected:
        raise RuntimeError(
            f"{row['run_name']} requested checkpoint is epoch {epoch}; expected epoch {expected}"
        )
    return path


def _require_final_checkpoint(row: dict[str, Any]) -> Path:
    path = _require_pinned_checkpoint(row)
    latest_path, latest_epoch = _latest_checkpoint(Path(row["checkpoint_dir"]))
    expected = int(row.get("checkpoint_epoch", int(row["epochs"]) - 1))
    if latest_epoch != expected:
        raise RuntimeError(
            f"{row['run_name']} latest checkpoint is {latest_path.name}; expected epoch {expected}"
        )
    return path


def _load_rows(project_dir: Path) -> list[dict[str, Any]]:
    path = project_dir / "local" / SWEEP_NAME / "manifest.json"
    rows = json.loads(path.read_text())
    sizes = tuple(sorted(int(row["dataset_size"]) for row in rows))
    if sizes != ALL_DATASET_SIZES:
        raise ValueError(f"Manifest sizes are {sizes}, expected {ALL_DATASET_SIZES}")
    if len({row["run_name"] for row in rows}) != len(ALL_DATASET_SIZES):
        raise ValueError("Manifest run names are not unique")
    return sorted(rows, key=lambda row: int(row["dataset_size"]))


def audit_prepared(project_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        size = int(row["dataset_size"])
        config_path = project_dir / row["config"]
        config = yaml.safe_load(config_path.read_text())
        if row.get("reuse_existing_checkpoint"):
            raise ValueError(f"Generator checkpoint reuse is forbidden for N={size}")
        if row.get("source_run_name") not in (None, ""):
            raise ValueError(f"Fresh run N={size} unexpectedly names a source run")
        if row.get("initialization") != "fresh":
            raise ValueError(f"Run N={size} is not marked as a fresh initialization")
        if int(row.get("training_seed", -1)) != TRAINING_SEED:
            raise ValueError(f"Run N={size} does not use training seed {TRAINING_SEED}")
        checkpoint_dir = Path(row["checkpoint_dir"])
        try:
            checkpoint_dir.relative_to(Path(CHECKPOINT_ROOT))
        except ValueError as exc:
            raise ValueError(
                f"Run N={size} checkpoint directory is outside the isolated fresh-sweep root: "
                f"{checkpoint_dir}"
            ) from exc
        if Path(config["io"]["output_dir"]) != checkpoint_dir:
            raise ValueError(f"Run N={size} config and manifest checkpoint directories differ")
        if config["model"]["kwargs"].get("encoder_hid_dim") != len(PARAM_NAMES):
            raise ValueError(f"{row['run_name']} is not conditioned on all six parameters")
        if config["train"].get("conditioning") != "continuous":
            raise ValueError(f"{row['run_name']} does not use continuous conditioning")
        for key in ("prepared_image_path", "train_label_path", "train_raw_params_path", "selected_pairs_path"):
            if not Path(row[key]).exists():
                raise FileNotFoundError(f"{row['run_name']} missing {key}: {row[key]}")
    print(
        "PREPARED PASS: ten fresh configs, exact data arrays, six-dimensional conditioning, "
        "isolated checkpoint paths, and fixed RNG seed."
    )


def audit_complete(project_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        checkpoint = _require_final_checkpoint(row)
        sample = project_dir / str(row["sample_path"]).format(seed=123, k=64)
        if not sample.exists():
            raise FileNotFoundError(f"Missing sample for N={row['dataset_size']}: {sample}")
        with np.load(sample, allow_pickle=True) as payload:
            sampled_epoch = int(payload["checkpoint_epoch"])
            sampled_checkpoint = str(payload["requested_checkpoint"])
        if sampled_epoch != int(row["checkpoint_epoch"]):
            raise ValueError(
                f"Sample for N={row['dataset_size']} used epoch {sampled_epoch}, "
                f"expected {row['checkpoint_epoch']}"
            )
        if sampled_checkpoint != str(row["requested_checkpoint"]):
            raise ValueError(
                f"Sample for N={row['dataset_size']} checkpoint provenance does not match manifest"
            )
        print(f"COMPLETE N={row['dataset_size']}: checkpoint={checkpoint.name}, sample={sample.name}")

    calibration = project_dir / "results" / SWEEP_NAME / "calibration_vgg"
    points_path = calibration / "bias_probe_per_cosmology_points.csv"
    slopes_path = calibration / "bias_probe_regime_slopes.csv"
    expected_figures = (
        calibration / "bias_probe_omega_m_all_dataset_sizes.png",
        calibration / "bias_probe_omega_m_transition_vs_dataset_size.png",
        calibration / "bias_probe_all_parameter_slopes_vs_dataset_size.png",
    )
    for path in (points_path, slopes_path, *expected_figures):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    points = pd.read_csv(points_path)
    slopes = pd.read_csv(slopes_path)
    for name, frame in (("points", points), ("slopes", slopes)):
        sizes = {int(value) for value in frame["dataset_size"].unique()}
        if sizes != set(ALL_DATASET_SIZES):
            raise ValueError(f"{name} table does not contain all ten dataset sizes")
        parameters = set(frame["parameter"].unique())
        if parameters != set(PARAM_NAMES):
            raise ValueError(f"{name} table parameter set is {sorted(parameters)}")
    print("COMPLETE PASS: all checkpoints, samples, six-parameter tables, and figures exist.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()
    rows = _load_rows(project_dir)
    audit_prepared(project_dir, rows)
    if args.complete:
        audit_complete(project_dir, rows)


if __name__ == "__main__":
    main()
