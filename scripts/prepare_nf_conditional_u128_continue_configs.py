#!/usr/bin/env python
"""Prepare continuation configs for the u128 conditional CAMELS runs.

The configs written here intentionally reuse the original checkpoint
directories:

- ``nf_conditional_u128`` for the continuous cosmology-parameter model
- ``nf_class_conditional_u128`` for the discrete field-class model

``cosmodiff_train.py`` resumes from the latest checkpoint in ``io.output_dir``.
In the Great Lakes runs used here, ``train.num_epochs`` is interpreted as the
number of epochs to execute after resuming, so this script sets it from the
requested additional optimizer updates rather than from an absolute final epoch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

import prepare_nf_class_conditional_u128_config as cls_base
import prepare_nf_conditional_u128_config as cont_base


CONTINUE_SWEEP_NAME = "nf_conditional_u128_continue400k"
ADDITIONAL_UPDATES = 400_000
CHECKPOINT_EVERY_UPDATES = 20_000


def ceil_updates_to_epochs(steps_per_epoch: int, updates: int) -> int:
    return max(1, math.ceil(int(updates) / int(steps_per_epoch)))


def checkpoint_epochs_for(steps_per_epoch: int) -> int:
    return max(1, round(CHECKPOINT_EVERY_UPDATES / int(steps_per_epoch)))


def continuous_label_paths(project_dir: Path, n_train_sims: int, sample_n: int) -> dict[str, Any]:
    name = cont_base.run_name(n_train_sims)
    label_dir = project_dir / "local" / cont_base.SWEEP_NAME / "labels"
    return {
        "train_label_path": label_dir / f"{name}_train_params_norm.npy",
        "train_raw_path": label_dir / f"{name}_train_params_raw.npy",
        "sample_label_path": label_dir / f"{name}_sample_params_norm_n{sample_n}.npy",
        "sample_raw_path": label_dir / f"{name}_sample_params_raw_n{sample_n}.npy",
        "sample_index_path": label_dir / f"{name}_sample_param_indices_n{sample_n}.txt",
        "stats_path": label_dir / f"{name}_param_norm_stats.json",
        "stats": {},
    }


def class_label_paths(
    project_dir: Path,
    fields: list[str],
    n_train_sims: int,
    sample_n: int,
) -> dict[str, Any]:
    name = cls_base.run_name(fields, n_train_sims)
    label_dir = project_dir / "local" / cls_base.SWEEP_NAME / "labels"
    class_map = {field: i for i, field in enumerate(fields)}
    return {
        "class_map": class_map,
        "train_label_paths": [
            label_dir / f"{name}_{field}_train_class.npy"
            for field in fields
        ],
        "sample_label_path": label_dir / f"{name}_sample_class_labels_n{sample_n}.npy",
        "sample_counts_path": label_dir / f"{name}_sample_class_counts_n{sample_n}.json",
        "class_map_path": label_dir / f"{name}_class_map.json",
    }


def update_row_for_continuation(row: dict[str, Any], *, additional_updates: int) -> dict[str, Any]:
    steps_per_epoch = int(row["steps_per_epoch"])
    continue_epochs = ceil_updates_to_epochs(steps_per_epoch, additional_updates)
    continue_actual_updates = continue_epochs * steps_per_epoch
    updated = dict(row)
    updated.update(
        {
            "continue_sweep_name": CONTINUE_SWEEP_NAME,
            "base_sweep_name": (
                cls_base.SWEEP_NAME
                if row["conditioning"] == "discrete"
                else cont_base.SWEEP_NAME
            ),
            "base_epochs": int(row["epochs"]),
            "base_target_updates": int(row["target_updates"]),
            "additional_target_updates": int(additional_updates),
            "continue_epochs": int(continue_epochs),
            "continue_actual_updates": int(continue_actual_updates),
            "epochs": int(continue_epochs),
            "checkpoint_every_updates": CHECKPOINT_EVERY_UPDATES,
            "checkpoint_every_n_epochs": int(checkpoint_epochs_for(steps_per_epoch)),
            "config": f"local/{CONTINUE_SWEEP_NAME}/configs/{row['run_name']}.yaml",
            "note": (
                "Continuation config: reuse the original checkpoint directory and "
                f"run about {additional_updates} additional optimizer updates."
            ),
        }
    )
    return updated


def build_continuous(
    args: argparse.Namespace,
    project_dir: Path,
    runtime_project_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.continuous_checkpoint_root)
    label_paths = continuous_label_paths(runtime_project_dir, args.n_train_sims, args.sample_n)
    config = cont_base.build_config(
        project_dir=project_dir,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        label_paths=label_paths,
    )
    row = cont_base.manifest_row(
        project_dir=project_dir,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        label_paths=label_paths,
    )
    row = update_row_for_continuation(row, additional_updates=args.additional_updates)
    config["train"]["num_epochs"] = int(row["continue_epochs"])
    config["train"]["checkpoint_every_n_epochs"] = int(row["checkpoint_every_n_epochs"])
    return config, row


def build_class(
    args: argparse.Namespace,
    project_dir: Path,
    runtime_project_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.class_checkpoint_root)
    fields = cls_base.parse_fields(args.fields)
    label_paths = class_label_paths(runtime_project_dir, fields, args.n_train_sims, args.sample_n)
    config = cls_base.build_config(
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        fields=fields,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        label_paths=label_paths,
    )
    row = cls_base.manifest_row(
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        fields=fields,
        n_train_sims=args.n_train_sims,
        sample_n=args.sample_n,
        label_paths=label_paths,
    )
    row = update_row_for_continuation(row, additional_updates=args.additional_updates)
    config["train"]["num_epochs"] = int(row["continue_epochs"])
    config["train"]["checkpoint_every_n_epochs"] = int(row["checkpoint_every_n_epochs"])
    return config, row


def selected_items(args: argparse.Namespace, project_dir: Path) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    runtime_project_dir = Path(args.runtime_project_dir) if args.runtime_project_dir else project_dir
    items = [
        ("continuous", *build_continuous(args, project_dir, runtime_project_dir)),
        ("class", *build_class(args, project_dir, runtime_project_dir)),
    ]
    if args.run_kind:
        allowed = set(args.run_kind)
        items = [item for item in items if item[0] in allowed]
    return items


def assert_config(path: Path, row: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)

    checks = {
        "io.output_dir": config["io"]["output_dir"] == row["checkpoint_dir"],
        "train.num_epochs": config["train"].get("num_epochs") == row["continue_epochs"],
        "train.checkpoint_every_n_epochs": (
            config["train"].get("checkpoint_every_n_epochs") == row["checkpoint_every_n_epochs"]
        ),
        "train.conditioning": config["train"].get("conditioning") == row["conditioning"],
        "generate.conditioning": config["generate"].get("conditioning") == row["conditioning"],
    }
    if row["conditioning"] == "continuous":
        checks.update(
            {
                "model.class": config["model"]["class"] == "UNet2DConditionModel",
                "model.encoder_hid_dim": config["model"]["kwargs"].get("encoder_hid_dim") == len(cont_base.PARAM_NAMES),
                "data.label_path": config["data"].get("label_path") == row["train_label_path"],
            }
        )
    else:
        checks.update(
            {
                "model.class": config["model"]["class"] == "UNet2DModel",
                "model.num_class_embeds": config["model"]["kwargs"].get("num_class_embeds") == row["num_class_embeds"],
                "data.label_path": config["data"].get("label_path") == row["train_label_paths"],
            }
        )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument(
        "--runtime-project-dir",
        default=None,
        help=(
            "Repository root to embed inside YAML paths. Defaults to --project-dir. "
            "Useful for materializing configs locally with Great Lakes /home paths."
        ),
    )
    parser.add_argument("--data-root", default=cont_base.DATA_ROOT, help="CAMELS 3d_grids directory.")
    parser.add_argument("--continuous-checkpoint-root", default=cont_base.CHECKPOINT_ROOT)
    parser.add_argument("--class-checkpoint-root", default=cls_base.CHECKPOINT_ROOT)
    parser.add_argument("--fields", default=",".join(cls_base.FIELDS), help="Comma-separated class fields.")
    parser.add_argument("--n-train-sims", type=int, default=cont_base.N_TRAIN_SIMS)
    parser.add_argument("--sample-n", type=int, default=cont_base.SAMPLE_N)
    parser.add_argument("--additional-updates", type=int, default=ADDITIONAL_UPDATES)
    parser.add_argument("--run-kind", choices=["continuous", "class"], action="append")
    parser.add_argument("--check-only", action="store_true", help="Validate selected continuation configs.")
    parser.add_argument("--print-runs", action="store_true", help="Print '<kind>\\t<run_name>' rows and exit.")
    parser.add_argument("--print-table", action="store_true", help="Print selected continuation rows and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_dir = project_dir / "local" / CONTINUE_SWEEP_NAME / "configs"
    manifest_path = project_dir / "local" / CONTINUE_SWEEP_NAME / "manifest.json"
    items = selected_items(args, project_dir)

    if args.print_runs:
        for kind, _config, row in items:
            print(f"{kind}\t{row['run_name']}")
        return

    if args.print_table:
        columns = [
            "kind",
            "run_name",
            "conditioning",
            "dataset_size",
            "steps_per_epoch",
            "base_epochs",
            "continue_epochs",
            "additional_target_updates",
            "continue_actual_updates",
            "checkpoint_every_n_epochs",
        ]
        print("\t".join(columns))
        for kind, _config, row in items:
            values = {"kind": kind, **row}
            print("\t".join(str(values[col]) for col in columns))
        return

    if args.check_only:
        for _kind, _config, row in items:
            assert_config(config_dir / f"{row['run_name']}.yaml", row)
        print(f"Validated {len(items)} {CONTINUE_SWEEP_NAME} configs.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, config, row in items:
        path = config_dir / f"{row['run_name']}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"Wrote {path}")
        rows.append({"kind": kind, **row})

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
