#!/usr/bin/env python
"""Prepare inference-only sampler configs for the u128 class-conditional run.

The training config stays DDPM/v-prediction.  These derived configs only set
the ``generate`` sampler fields used by ``cosmodiff_sample.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import prepare_nf_class_conditional_u128_config as base


COMPARISON_SAMPLE_N = 96
FULL_SAMPLE_N = base.SAMPLE_N
SAMPLE_BATCH_SIZE = 8
COMPARISON_SAMPLERS = (
    ("ddpm500", "DDPMScheduler", 500),
    ("dpm50", "DPMSolverMultistepScheduler", 50),
)
FULL_SAMPLERS = (
    ("dpm50", "DPMSolverMultistepScheduler", 50),
)


def _write_base_training_files(
    *,
    project_dir: Path,
    data_root: Path,
    checkpoint_root: Path,
    fields: list[str],
    n_train_sims: int,
    sample_n: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label_paths = base.write_labels(
        project_dir=project_dir,
        fields=fields,
        n_train_sims=n_train_sims,
        sample_n=sample_n,
    )
    config = base.build_config(
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        fields=fields,
        n_train_sims=n_train_sims,
        sample_n=sample_n,
        label_paths=label_paths,
    )
    row = base.manifest_row(
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        fields=fields,
        n_train_sims=n_train_sims,
        sample_n=sample_n,
        label_paths=label_paths,
    )

    name = base.run_name(fields, n_train_sims)
    config_path = project_dir / "local" / base.SWEEP_NAME / "configs" / f"{name}.yaml"
    manifest_path = project_dir / "local" / base.SWEEP_NAME / "manifest.json"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump([row], f, indent=2)
        f.write("\n")

    return config, row


def _sampling_labels(
    *,
    project_dir: Path,
    fields: list[str],
    n_train_sims: int,
    sample_n: int,
) -> dict[str, Any]:
    return base.write_labels(
        project_dir=project_dir,
        fields=fields,
        n_train_sims=n_train_sims,
        sample_n=sample_n,
    )


def _build_sampling_config(
    *,
    training_config: dict[str, Any],
    scheduler: str,
    num_steps: int,
    sample_n: int,
    batch_size: int,
    labels_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(training_config)
    generate = config.setdefault("generate", {})
    generate.update(
        {
            "scheduler": scheduler,
            "num_steps": int(num_steps),
            "n_samples": int(sample_n),
            "batch_size": int(batch_size),
            "conditioning": "discrete",
            "labels": str(labels_path),
            "continuous_labels": None,
        }
    )
    return config


def _rows_for_stage(
    *,
    project_dir: Path,
    run_name: str,
    stage: str,
    sample_n: int,
    batch_size: int,
    labels_path: Path,
    samplers: tuple[tuple[str, str, int], ...],
) -> list[dict[str, Any]]:
    rows = []
    for sampler_label, scheduler, num_steps in samplers:
        if stage == "comparison":
            sample_rel = (
                f"results/{base.SWEEP_NAME}/sampler_compare/"
                f"{run_name}_seed{{seed}}_{sampler_label}_n{sample_n}.npz"
            )
        else:
            sample_rel = (
                f"results/{base.SWEEP_NAME}/samples/"
                f"{run_name}_seed{{seed}}_{sampler_label}_class_conditional.npz"
            )
        config_rel = (
            f"local/{base.SWEEP_NAME}/sampling_configs/"
            f"{run_name}_{sampler_label}_n{sample_n}.yaml"
        )
        rows.append(
            {
                "stage": stage,
                "run_name": run_name,
                "sampler_label": sampler_label,
                "scheduler": scheduler,
                "num_steps": int(num_steps),
                "n_samples": int(sample_n),
                "batch_size": int(batch_size),
                "labels": str(labels_path),
                "config": config_rel,
                "sample_path": sample_rel,
                "inference_only": True,
            }
        )
    return rows


def _assert_sampling_config(path: Path, row: dict[str, Any]) -> None:
    with path.open() as f:
        config = yaml.safe_load(f)
    failed = []
    checks = {
        "noise_scheduler.class": config["noise_scheduler"]["class"] == "DDPMScheduler",
        "generate.scheduler": config["generate"].get("scheduler") == row["scheduler"],
        "generate.num_steps": config["generate"].get("num_steps") == row["num_steps"],
        "generate.n_samples": config["generate"].get("n_samples") == row["n_samples"],
        "generate.labels": config["generate"].get("labels") == row["labels"],
        "generate.conditioning": config["generate"].get("conditioning") == "discrete",
    }
    failed.extend(name for name, ok in checks.items() if not ok)
    labels = np.load(row["labels"])
    if labels.shape != (row["n_samples"],):
        failed.append("labels.shape")
    if not np.issubdtype(labels.dtype, np.integer):
        failed.append("labels.integer")
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def write_sampling_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    project_dir = Path(args.project_dir).resolve()
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    fields = base.parse_fields(args.fields)
    name = base.run_name(fields, args.n_train_sims)

    training_config, _training_row = _write_base_training_files(
        project_dir=project_dir,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        fields=fields,
        n_train_sims=args.n_train_sims,
        sample_n=args.full_sample_n,
    )

    out_dir = project_dir / "local" / base.SWEEP_NAME / "sampling_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    for stage, sample_n, samplers in (
        ("comparison", args.comparison_sample_n, COMPARISON_SAMPLERS),
        ("full", args.full_sample_n, FULL_SAMPLERS),
    ):
        label_paths = _sampling_labels(
            project_dir=project_dir,
            fields=fields,
            n_train_sims=args.n_train_sims,
            sample_n=sample_n,
        )
        rows = _rows_for_stage(
            project_dir=project_dir,
            run_name=name,
            stage=stage,
            sample_n=sample_n,
            batch_size=args.batch_size,
            labels_path=label_paths["sample_label_path"],
            samplers=samplers,
        )
        for row in rows:
            config = _build_sampling_config(
                training_config=training_config,
                scheduler=row["scheduler"],
                num_steps=row["num_steps"],
                sample_n=row["n_samples"],
                batch_size=row["batch_size"],
                labels_path=Path(row["labels"]),
            )
            config_path = project_dir / row["config"]
            with config_path.open("w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
            _assert_sampling_config(config_path, row)
            manifest_rows.append(row)
            print(f"Wrote {config_path}")

    manifest_path = project_dir / "local" / base.SWEEP_NAME / "sampling_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest_rows, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--data-root", default=base.DATA_ROOT)
    parser.add_argument("--checkpoint-root", default=base.CHECKPOINT_ROOT)
    parser.add_argument("--fields", default=",".join(base.FIELDS))
    parser.add_argument("--n-train-sims", type=int, default=base.N_TRAIN_SIMS)
    parser.add_argument("--comparison-sample-n", type=int, default=COMPARISON_SAMPLE_N)
    parser.add_argument("--full-sample-n", type=int, default=FULL_SAMPLE_N)
    parser.add_argument("--batch-size", type=int, default=SAMPLE_BATCH_SIZE)
    parser.add_argument("--print-table", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_sampling_configs(args)
    if args.print_table:
        cols = ["stage", "sampler_label", "scheduler", "num_steps", "n_samples", "config"]
        print("\t".join(cols))
        for row in rows:
            print("\t".join(str(row[col]) for col in cols))


if __name__ == "__main__":
    main()
