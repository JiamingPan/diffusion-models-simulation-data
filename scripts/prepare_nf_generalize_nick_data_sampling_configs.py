#!/usr/bin/env python
"""Prepare inference-only DPM-Solver configs for nf_generalize_nick_data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

import prepare_nf_generalize_nick_data_configs as base


DEFAULT_SAMPLE_LABEL = "dpm50"
DEFAULT_SCHEDULER = "DPMSolverMultistepScheduler"
DEFAULT_NUM_STEPS = 50
DEFAULT_SAMPLE_N = 512
DEFAULT_BATCH_SIZE = 8


def selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = base.iter_runs()
    if args.run_name:
        wanted = set(args.run_name)
        rows = [row for row in rows if row["run_name"] in wanted]
    if args.dataset_tag:
        wanted = set(args.dataset_tag)
        rows = [row for row in rows if row["dataset_tag"] in wanted]
    return rows


def build_sampling_config(
    config: dict[str, Any],
    *,
    scheduler: str,
    num_steps: int,
    sample_n: int,
    batch_size: int,
) -> dict[str, Any]:
    out = copy.deepcopy(config)
    generate = out.setdefault("generate", {})
    generate.update(
        {
            "scheduler": scheduler,
            "num_steps": int(num_steps),
            "n_samples": int(sample_n),
            "batch_size": int(batch_size),
            "conditioning": "discrete",
            "labels": None,
            "continuous_labels": None,
        }
    )
    return out


def assert_sampling_config(path: Path, row: dict[str, Any]) -> None:
    with path.open() as f:
        config = yaml.safe_load(f)
    failed = []
    checks = {
        "noise_scheduler.class": config["noise_scheduler"]["class"] == "DDPMScheduler",
        "noise_scheduler.timesteps": config["noise_scheduler"]["kwargs"].get("num_train_timesteps") == 500,
        "generate.scheduler": config["generate"].get("scheduler") == row["scheduler"],
        "generate.num_steps": config["generate"].get("num_steps") == row["num_steps"],
        "generate.n_samples": config["generate"].get("n_samples") == row["n_samples"],
        "generate.batch_size": config["generate"].get("batch_size") == row["batch_size"],
    }
    failed.extend(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def write_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    project_dir = Path(args.project_dir).resolve()
    config_root = project_dir / "local" / base.SWEEP_NAME / "configs"
    out_root = project_dir / "local" / base.SWEEP_NAME / "sampling_configs"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = selected_rows(args)
    if not rows:
        raise SystemExit("No nf_generalize_nick_data rows selected.")

    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        source_path = config_root / f"{row['run_name']}.yaml"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing base config: {source_path}")
        with source_path.open() as f:
            config = yaml.safe_load(f)

        sample_label = args.sample_label
        out_path = out_root / f"{row['run_name']}_{sample_label}.yaml"
        sampling_config = build_sampling_config(
            config,
            scheduler=args.scheduler,
            num_steps=args.num_steps,
            sample_n=args.sample_n,
            batch_size=args.batch_size,
        )
        with out_path.open("w") as f:
            yaml.safe_dump(sampling_config, f, sort_keys=False)

        out_row = {
            "run_name": row["run_name"],
            "dataset_tag": row["dataset_tag"],
            "dataset_size": int(row["dataset_size"]),
            "sample_label": sample_label,
            "scheduler": args.scheduler,
            "num_steps": int(args.num_steps),
            "n_samples": int(args.sample_n),
            "batch_size": int(args.batch_size),
            "config": str(out_path.relative_to(project_dir)),
            "checkpoint_dir": row["checkpoint_dir"],
            "ddpm_reference_sample_path": row["sample_path"],
            "sample_path": f"results/{base.SWEEP_NAME}/samples/{row['run_name']}_seed{{seed}}_{sample_label}.npz",
            "inference_only": True,
        }
        assert_sampling_config(out_path, out_row)
        manifest_rows.append(out_row)
        print(f"Wrote {out_path}")

    manifest_path = project_dir / "local" / base.SWEEP_NAME / f"sampling_manifest_{args.sample_label}.json"
    with manifest_path.open("w") as f:
        json.dump(manifest_rows, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--dataset-tag", action="append")
    parser.add_argument("--sample-label", default=DEFAULT_SAMPLE_LABEL)
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--sample-n", type=int, default=DEFAULT_SAMPLE_N)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--print-table", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_configs(args)
    if args.print_table:
        cols = ["run_name", "dataset_tag", "sample_label", "scheduler", "num_steps", "n_samples", "config"]
        print("\t".join(cols))
        for row in rows:
            print("\t".join(str(row[col]) for col in cols))


if __name__ == "__main__":
    main()
