#!/usr/bin/env python
"""Fail-closed state audit for the two DiT-L16 seed-restart runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_nf_generalize_fig2_dit_l16_seed_restart500k_configs as prep
from simdiff_eval.terminal_reports import start_report
from run_cosmodiff_train_with_dit_resume import (
    normalize_posthoc_ema_checkpoint_state,
)


CHECKPOINT_RE = re.compile(r"checkpoint-epoch-(\d+)$")
REQUIRED_STATE = {
    "model": (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    ),
    "model_config": ("config.json",),
    "checkpoint_config": ("checkpoint_config.yaml",),
    "noise_scheduler": ("scheduler_config.json", "noise_scheduler.pkl"),
    "scaler": ("scaler.pt", "scaler.bin"),
    "rng": ("random_states_0.pkl", "random_states.pkl"),
}


def _training_state_layout(checkpoint: Path) -> tuple[str, Path, Path] | None:
    legacy = (
        checkpoint / "optimizer.pkl",
        checkpoint / "lr_scheduler.pkl",
    )
    native = (
        checkpoint / "optimizer.bin",
        checkpoint / "scheduler.bin",
    )
    if all(path.is_file() for path in legacy):
        return "legacy", *legacy
    if all(path.is_file() for path in native):
        return "native", *native
    return None


def _state_file(checkpoint: Path, alternatives: tuple[str, ...]) -> Path | None:
    for name in alternatives:
        path = checkpoint / name
        if path.is_file():
            return path
    return None


def _checkpoint_epoch(checkpoint: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(Path(checkpoint).name)
    if match is None:
        raise ValueError(f"invalid checkpoint name: {checkpoint}")
    return int(match.group(1))


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_seed_restart_row(
    row: dict[str, Any], project_dir: Path
) -> dict[str, Any]:
    """Validate the exact checkpoint that one stage will resume."""
    project_dir = Path(project_dir).resolve()
    if row.get("dataset_tag") not in prep.EXPECTED_TAGS:
        raise ValueError(f"unexpected dataset tag: {row.get('dataset_tag')}")
    if int(row.get("resume_seed", -1)) != prep.RESUME_SEED:
        raise ValueError("resume seed mismatch")

    source_config = prep._project_path(project_dir, row["source_config"])
    config_path = prep._project_path(project_dir, row["config"])
    if prep.sha256_file(source_config) != row["source_config_sha256"]:
        raise ValueError(f"source config digest changed: {source_config}")
    prep.assert_seed_restart_config(source_config, config_path, row)
    config = yaml.safe_load(config_path.read_text())
    train = config["train"]
    if int(train.get("ema_update_every", -1)) != 1:
        raise ValueError("seed restart requires ema_update_every=1")
    accumulation = int(train.get("gradient_accumulation_steps", 1))
    if accumulation != int(row["microbatches_per_optimizer_step"]):
        raise ValueError(
            "gradient accumulation mismatch: "
            f"config={accumulation}, manifest={row['microbatches_per_optimizer_step']}"
        )
    ema_burn_in = int(train["ema_burn_in"])
    profile_count = len(train["ema_sigma_rels"])
    if profile_count < 2:
        raise ValueError("seed restart requires at least two EMA profiles")

    checkpoint = Path(row["previous_expected_checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"resume checkpoint: missing {checkpoint}")
    state_files: dict[str, str] = {}
    state_layout = _training_state_layout(checkpoint)
    if state_layout is None:
        raise FileNotFoundError(
            f"optimizer/LR scheduler: {checkpoint} lacks a coherent native or legacy pair"
        )
    layout_name, optimizer_path, lr_scheduler_path = state_layout
    state_files["optimizer"] = str(optimizer_path)
    state_files["lr_scheduler"] = str(lr_scheduler_path)
    for group, alternatives in REQUIRED_STATE.items():
        path = _state_file(checkpoint, alternatives)
        if path is None:
            raise FileNotFoundError(
                f"{group}: {checkpoint} contains none of {', '.join(alternatives)}"
            )
        state_files[group] = str(path)

    model_config = json.loads((checkpoint / "config.json").read_text())
    class_name = str(model_config.get("_class_name", "")).rsplit(".", 1)[-1]
    if class_name != "DiTTransformer2DModel":
        raise ValueError(f"checkpoint class is {class_name}, not DiTTransformer2DModel")

    checkpoint_config = yaml.safe_load(
        (checkpoint / "checkpoint_config.yaml").read_text()
    )
    checkpoint_sigma_rels = [
        float(value) for value in checkpoint_config.get("ema_sigma_rels") or []
    ]
    expected_sigma_rels = [float(value) for value in train["ema_sigma_rels"]]
    if checkpoint_sigma_rels != expected_sigma_rels:
        raise ValueError(
            "Checkpoint EMA sigma profiles do not match continuation config: "
            f"checkpoint={checkpoint_sigma_rels}, expected={expected_sigma_rels}"
        )
    checkpoint_burn_in = int(checkpoint_config.get("ema_burn_in", -1))
    if checkpoint_burn_in != ema_burn_in:
        raise ValueError(
            "Checkpoint EMA burn-in does not match continuation config: "
            f"checkpoint={checkpoint_burn_in}, expected={ema_burn_in}"
        )

    epoch = _checkpoint_epoch(checkpoint)
    steps_per_epoch = int(row["optimizer_steps_per_epoch"])
    source_updates = (epoch + 1) * steps_per_epoch
    expected_updates = int(row["previous_actual_total_updates"])
    if source_updates != expected_updates:
        raise ValueError(
            f"resume optimizer clock mismatch: checkpoint={source_updates}, "
            f"manifest={expected_updates}"
        )
    source_microbatches = source_updates * accumulation
    expected_microbatches = int(row["previous_total_microbatches"])
    if source_microbatches != expected_microbatches:
        raise ValueError(
            f"resume microbatch clock mismatch: checkpoint={source_microbatches}, "
            f"manifest={expected_microbatches}"
        )
    expected_ema_step = int(row["previous_expected_ema_step"])
    if expected_ema_step <= 0:
        raise ValueError("resume checkpoint precedes the EMA burn-in")
    ema_snapshots: list[str] = []
    for profile_index in range(profile_count):
        path = checkpoint / "ema" / f"{profile_index}.{expected_ema_step}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"ema profile: missing {path}")
        state = _load_torch(path)
        if not isinstance(state, dict):
            raise ValueError(f"EMA snapshot is not a state mapping: {path}")
        state = normalize_posthoc_ema_checkpoint_state(
            state,
            expected_step=expected_ema_step,
        )
        actual_step = int(state["step"].item())
        if actual_step != expected_ema_step:
            raise ValueError(
                f"EMA snapshot {path} records step {actual_step}, "
                f"expected {expected_ema_step}"
            )
        if not bool(state["initted"].item()):
            raise ValueError(f"EMA snapshot is not initialized: {path}")
        if not any(str(key).startswith("ema_model.") for key in state):
            raise ValueError(f"EMA snapshot has no ema_model weights: {path}")
        ema_snapshots.append(str(path))

    byte_identical = None
    if int(row["continue_stage"]) == 1:
        source_checkpoint = Path(row["source_checkpoint"])
        byte_identical = (
            prep.checkpoint_inventory(source_checkpoint)
            == prep.checkpoint_inventory(checkpoint)
        )
        if not byte_identical:
            raise ValueError(
                f"seed checkpoint is not byte-identical to source: {checkpoint}"
            )

    return {
        "dataset_tag": row["dataset_tag"],
        "continue_stage": int(row["continue_stage"]),
        "checkpoint": str(checkpoint),
        "source_updates": source_updates,
        "source_microbatches": source_microbatches,
        "microbatches_per_optimizer_step": accumulation,
        "expected_ema_step": expected_ema_step,
        "ema_snapshots": ema_snapshots,
        "state_files": state_files,
        "training_state_layout": layout_name,
        "seed_checkpoint_byte_identical": byte_identical,
        "resume_seed": prep.RESUME_SEED,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--dataset-tag", action="append")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    manifest = args.manifest or (
        project_dir / "local" / prep.SWEEP_NAME / "manifest.json"
    )
    if not manifest.is_absolute():
        manifest = project_dir / manifest
    rows = json.loads(manifest.read_text())
    selected = prep.selected_rows(
        rows,
        dataset_tags=args.dataset_tag,
        stages=[args.stage],
    )
    expected_count = len(args.dataset_tag or prep.EXPECTED_TAGS)
    if len(selected) != expected_count:
        raise ValueError(
            f"stage {args.stage} requires {expected_count} rows; found {len(selected)}"
        )
    report = {
        "sweep_name": prep.SWEEP_NAME,
        "resume_seed": prep.RESUME_SEED,
        "stage": int(args.stage),
        "rows": [validate_seed_restart_row(row, project_dir) for row in selected],
    }
    if args.report:
        report_path = args.report
        if not report_path.is_absolute():
            report_path = project_dir / report_path
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite preflight report: {report_path}")
        report = start_report(
            report_path,
            payload=report,
            producer_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        print(f"Wrote {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
