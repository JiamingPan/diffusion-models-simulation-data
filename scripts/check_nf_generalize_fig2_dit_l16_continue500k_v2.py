#!/usr/bin/env python
"""Validate every source and architecture contract for the 300k-to-500k sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs as prep


EXPECTED_DIT_L16 = {
    "sample_size": 128,
    "patch_size": 8,
    "num_layers": 16,
    "num_attention_heads": 12,
    "attention_head_dim": 64,
    "norm_num_groups": 32,
}
EXPECTED_CLASS = "DiTTransformer2DModel"
REQUIRED_STATE_GROUPS = {
    "model": (
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
    ),
    "model_config": ("config.json",),
    "checkpoint_config": ("checkpoint_config.yaml",),
    "optimizer": ("optimizer.pkl",),
    "noise_scheduler": ("noise_scheduler.pkl",),
    "lr_scheduler": ("lr_scheduler.pkl",),
    "scaler": ("scaler.pt", "scaler.bin"),
    "rng": ("random_states_0.pkl", "random_states.pkl"),
    "ema": ("ema_state.pt", "ema.pt", "ema_model.pt", "ema/"),
}
ARCHITECTURE_ALLOWED_DIFFERENCES = {
    "io.output_dir",
    "model.num_layers",
    "model.kwargs.num_layers",
    "train.num_epochs",
    "train.checkpoint_every_n_epochs",
}


def _project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    kwargs = model.get("kwargs")
    return kwargs if isinstance(kwargs, dict) else model


def _model_class(config: dict[str, Any]) -> str | None:
    model = config.get("model", {})
    value = model.get("class") or model.get("_class_name")
    return str(value).rsplit(".", 1)[-1] if value else None


def _architecture(config: dict[str, Any]) -> dict[str, Any]:
    model = _model_kwargs(config)
    return {key: model.get(key) for key in EXPECTED_DIT_L16}


def _state_path(checkpoint: Path, alternatives: tuple[str, ...]) -> Path | None:
    for name in alternatives:
        path = checkpoint / name.rstrip("/")
        if name.endswith("/"):
            if path.is_dir() and any(item.is_file() for item in path.rglob("*")):
                return path
        elif path.is_file():
            return path
    return None


def _assert_checkpoint_config(checkpoint: Path) -> None:
    checkpoint_config = yaml.safe_load(
        (checkpoint / "checkpoint_config.yaml").read_text()
    )
    for section in ("noise_scheduler", "optimizer", "lr_scheduler"):
        if not isinstance(checkpoint_config.get(section), dict):
            raise ValueError(
                f"checkpoint_config: missing {section} section in {checkpoint}"
            )


def validate_source_row(
    row: dict[str, Any], project_dir: Path
) -> dict[str, Any]:
    """Validate one immutable 300k source without importing GPU libraries."""
    project_dir = Path(project_dir).resolve()
    if row.get("source_sweep_name") != prep.SOURCE_SWEEP_NAME:
        raise ValueError(
            f"{row.get('dataset_tag')}: source sweep is not {prep.SOURCE_SWEEP_NAME}"
        )
    if int(row.get("continue_stage", -1)) != 1:
        raise ValueError("validate_source_row requires a first-stage manifest row")

    source_config = _project_path(project_dir, row["source_config"])
    if not source_config.is_file():
        raise FileNotFoundError(f"source config: missing {source_config}")
    actual_digest = prep.sha256_file(source_config)
    if actual_digest != row.get("source_config_sha256"):
        raise ValueError(
            f"source config digest mismatch for {row.get('dataset_tag')}: "
            f"expected {row.get('source_config_sha256')}, found {actual_digest}"
        )
    config = yaml.safe_load(source_config.read_text())
    architecture = _architecture(config)
    mismatches = {
        key: (EXPECTED_DIT_L16[key], architecture.get(key))
        for key in EXPECTED_DIT_L16
        if architecture.get(key) != EXPECTED_DIT_L16[key]
    }
    if mismatches:
        details = ", ".join(
            f"{key}=expected {expected}, found {actual}"
            for key, (expected, actual) in mismatches.items()
        )
        raise ValueError(f"architecture contract mismatch: {details}")
    declared_class = _model_class(config)
    if declared_class not in (None, EXPECTED_CLASS):
        raise ValueError(
            f"model class must be {EXPECTED_CLASS}, found {declared_class}"
        )

    data = config.get("data", {})
    if data.get("augment") not in (None, False) or row.get("augmentations") not in (
        None,
        False,
        [],
    ):
        raise ValueError("augmentation must be disabled for the controlled sweep")
    constant_label = data.get("constant_label", data.get("class_label", 0))
    if constant_label != 0:
        raise ValueError(f"constant_label must be 0, found {constant_label}")

    checkpoint = Path(row["source_checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"source checkpoint: missing exact 300k checkpoint {checkpoint}"
        )
    resume_state: dict[str, str] = {}
    for group, alternatives in REQUIRED_STATE_GROUPS.items():
        path = _state_path(checkpoint, alternatives)
        if path is None:
            raise FileNotFoundError(
                f"{group}: {checkpoint} contains none of {', '.join(alternatives)}"
            )
        resume_state[group] = str(path)

    model_config = json.loads((checkpoint / "config.json").read_text())
    class_name = str(model_config.get("_class_name", "")).rsplit(".", 1)[-1]
    if class_name != EXPECTED_CLASS:
        raise ValueError(
            f"checkpoint model class must be {EXPECTED_CLASS}, found {class_name}"
        )
    for key, expected in EXPECTED_DIT_L16.items():
        actual = model_config.get(key)
        if actual is not None and actual != expected:
            raise ValueError(
                f"checkpoint {key}=expected {expected}, found {actual}: {checkpoint}"
            )
    _assert_checkpoint_config(checkpoint)

    return {
        "dataset_tag": row["dataset_tag"],
        "dataset_size": int(row["dataset_size"]),
        "source_config": str(source_config),
        "source_config_sha256": actual_digest,
        "source_checkpoint": str(checkpoint),
        "architecture": architecture,
        "constant_label": constant_label,
        "resume_state": resume_state,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], path))
        return flattened
    return {prefix: value}


def compare_architecture_configs(
    config_paths: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    required = ("dit_l8", "dit_l12", "dit_l16")
    if tuple(config_paths) != required and set(config_paths) != set(required):
        raise ValueError(f"architecture comparison requires {', '.join(required)}")
    configs = {
        name: yaml.safe_load(Path(config_paths[name]).read_text()) for name in required
    }
    flattened = {name: _flatten(config) for name, config in configs.items()}
    keys = set().union(*(values.keys() for values in flattened.values()))
    forbidden = []
    for key in sorted(keys):
        values = [flattened[name].get(key) for name in required]
        if len({json.dumps(value, sort_keys=True) for value in values}) > 1:
            if key not in ARCHITECTURE_ALLOWED_DIFFERENCES:
                forbidden.append(key)
    if forbidden:
        raise ValueError(
            "L8/L12/L16 configs differ outside depth/run fields: "
            + ", ".join(forbidden)
        )

    report: dict[str, dict[str, Any]] = {}
    for name in required:
        architecture = _architecture(configs[name])
        report[name] = {
            **architecture,
            "model_class": _model_class(configs[name]),
            "constant_label": configs[name].get("data", {}).get(
                "constant_label", 0
            ),
            "normalization": configs[name].get("data", {}).get("normalization"),
            "transform": configs[name].get("data", {}).get("transform"),
        }
    expected_depths = {"dit_l8": 8, "dit_l12": 12, "dit_l16": 16}
    for name, depth in expected_depths.items():
        if report[name]["num_layers"] != depth:
            raise ValueError(
                f"{name} num_layers expected {depth}, found {report[name]['num_layers']}"
            )
    return report


def _find_comparison_configs(
    project_dir: Path,
    first_rows: list[dict[str, Any]],
    dataset_tag: str,
) -> dict[str, Path]:
    base_manifest = project_dir / "local" / "nf_generalize_fig2_dit" / "manifest.json"
    if not base_manifest.is_file():
        raise FileNotFoundError(
            "architecture comparison requires the original DiT manifest: "
            f"{base_manifest}"
        )
    base_rows = json.loads(base_manifest.read_text())
    selected: dict[str, Path] = {}
    mapping = {"dit_l8": "dit_l8", "dit_base": "dit_l12"}
    for row in base_rows:
        if row.get("dataset_tag") == dataset_tag and row.get("arch") in mapping:
            selected[mapping[row["arch"]]] = _project_path(project_dir, row["config"])
    l16 = next(row for row in first_rows if row["dataset_tag"] == dataset_tag)
    selected["dit_l16"] = _project_path(project_dir, l16["source_config"])
    if set(selected) != {"dit_l8", "dit_l12", "dit_l16"}:
        raise ValueError(f"could not find complete comparison configs for {dataset_tag}")
    return {name: selected[name] for name in ("dit_l8", "dit_l12", "dit_l16")}


def _runtime_validate(
    project_dir: Path,
    checkpoints: list[Path],
    *,
    cosmodiff_dir: Path,
) -> None:
    checker = project_dir / "scripts" / "check_nf_generalize_fig2_dit_resume.py"
    for checkpoint in checkpoints:
        subprocess.run(
            [
                sys.executable,
                str(checker),
                "--checkpoint",
                str(checkpoint),
                "--cosmodiff-dir",
                str(cosmodiff_dir),
            ],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--comparison-dataset-tag", default="d2p10")
    parser.add_argument("--runtime-check", action="store_true")
    parser.add_argument(
        "--cosmodiff-dir",
        default="/home/jiamingp/Diffusion_model/cosmo_diffusion_main",
        type=Path,
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    manifest = args.manifest or prep.continuation_manifest_path(project_dir)
    if not manifest.is_absolute():
        manifest = project_dir / manifest
    rows = json.loads(manifest.read_text())
    prep._check_frozen_rows(project_dir, rows)
    first_rows = sorted(
        (row for row in rows if int(row["continue_stage"]) == 1),
        key=lambda row: int(row["dataset_size"]),
    )
    if len(first_rows) != 10:
        raise ValueError(f"expected ten first-stage rows, found {len(first_rows)}")

    source_reports = [validate_source_row(row, project_dir) for row in first_rows]
    comparison_paths = _find_comparison_configs(
        project_dir, first_rows, args.comparison_dataset_tag
    )
    architecture_report = compare_architecture_configs(comparison_paths)
    if args.runtime_check:
        _runtime_validate(
            project_dir,
            [Path(report["source_checkpoint"]) for report in source_reports],
            cosmodiff_dir=args.cosmodiff_dir.expanduser().resolve(),
        )

    report = {
        "sweep_name": prep.CONTINUE_SWEEP_NAME,
        "source_sweep_name": prep.SOURCE_SWEEP_NAME,
        "validated_source_count": len(source_reports),
        "runtime_check": bool(args.runtime_check),
        "source_rows": source_reports,
        "architecture_comparison_dataset_tag": args.comparison_dataset_tag,
        "architecture_comparison": architecture_report,
    }
    report_path = args.report or (
        project_dir
        / "local"
        / prep.CONTINUE_SWEEP_NAME
        / "precheck_report.json"
    )
    if not report_path.is_absolute():
        report_path = project_dir / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("dataset\tsize\tpatch\tdepth\theads\thead_dim\tnorm_groups")
    for source in source_reports:
        arch = source["architecture"]
        print(
            f"{source['dataset_tag']}\t{source['dataset_size']}\t"
            f"{arch['patch_size']}\t{arch['num_layers']}\t"
            f"{arch['num_attention_heads']}\t{arch['attention_head_dim']}\t"
            f"{arch['norm_num_groups']}"
        )
    print(json.dumps(architecture_report, indent=2))
    print(f"PASS: validated ten exact full-state 300k sources; wrote {report_path}")


if __name__ == "__main__":
    main()
