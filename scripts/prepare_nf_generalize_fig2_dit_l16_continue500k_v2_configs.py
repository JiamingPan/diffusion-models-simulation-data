#!/usr/bin/env python
"""Freeze a full-state DiT-L16 continuation sweep from 300k to 500k updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


SOURCE_SWEEP_NAME = "nf_generalize_fig2_dit_l16_fresh300k_v2"
CONTINUE_SWEEP_NAME = "nf_generalize_fig2_dit_l16_continue500k_v2"
MANIFEST_VERSION = 1
SOURCE_TARGET_UPDATES = 300_000
TARGET_UPDATES = (340_000, 380_000, 420_000, 460_000, 500_000)
STAGE_UPDATES = 40_000
RESTART_PERIOD_UPDATES = 4_000
CHECKPOINT_EVERY_TARGET_UPDATES = 5_000
EXPECTED_TAGS = tuple(f"d2p{power:02d}" for power in range(6, 16))
DEFAULT_CHECKPOINT_ROOT = Path(
    f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{CONTINUE_SWEEP_NAME}"
)
ALLOWED_CONFIG_CHANGES = {
    "io.output_dir",
    "train.num_epochs",
    "train.checkpoint_every_n_epochs",
}


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _stored_path(project_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_dir.resolve()))
    except ValueError:
        return str(path)


def continuation_manifest_path(project_dir: Path) -> Path:
    return project_dir / "local" / CONTINUE_SWEEP_NAME / "manifest.json"


def analysis_manifest_path(project_dir: Path) -> Path:
    return project_dir / "local" / CONTINUE_SWEEP_NAME / "analysis_manifest.json"


def source_manifest_path(project_dir: Path) -> Path:
    return project_dir / "local" / SOURCE_SWEEP_NAME / "manifest.json"


def _validate_source_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tags = [str(row.get("dataset_tag", "")) for row in source_rows]
    if sorted(tags) != sorted(EXPECTED_TAGS) or len(set(tags)) != len(EXPECTED_TAGS):
        raise ValueError(
            "source manifest dataset tags must contain exactly one each of "
            f"{', '.join(EXPECTED_TAGS)}; found {tags}"
        )
    for row in source_rows:
        if row.get("sweep_name") != SOURCE_SWEEP_NAME:
            raise ValueError(
                f"{row.get('dataset_tag')}: source sweep must be {SOURCE_SWEEP_NAME}"
            )
        if row.get("arch") != "dit_l16":
            raise ValueError(f"{row.get('dataset_tag')}: source architecture is not dit_l16")
        if int(row.get("target_total_updates", -1)) != SOURCE_TARGET_UPDATES:
            raise ValueError(
                f"{row.get('dataset_tag')}: source target is not {SOURCE_TARGET_UPDATES}"
            )
        if int(row.get("optimizer_steps_per_epoch", 0)) <= 0:
            raise ValueError(
                f"{row.get('dataset_tag')}: optimizer_steps_per_epoch must be positive"
            )
        for field in ("config", "expected_checkpoint", "sample_path"):
            if not row.get(field):
                raise ValueError(f"{row.get('dataset_tag')}: missing source field {field}")
    return sorted(source_rows, key=lambda row: int(row["dataset_size"]))


def _continuation_run_name(dataset_tag: str) -> str:
    return f"nf_fig2_dit_l16_{dataset_tag}_noaug_continue500k_v2_seed123"


def build_continuation_rows(
    project_dir: Path,
    source_rows: list[dict[str, Any]],
    *,
    target_updates: tuple[int, ...] = TARGET_UPDATES,
    checkpoint_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic stage rows without mutating source files."""
    project_dir = Path(project_dir).resolve()
    ordered_sources = _validate_source_rows(source_rows)
    targets = tuple(int(value) for value in target_updates)
    expected_targets = tuple(
        SOURCE_TARGET_UPDATES + STAGE_UPDATES * index
        for index in range(1, len(targets) + 1)
    )
    if targets != expected_targets:
        raise ValueError(
            f"continuation targets must be equally spaced by {STAGE_UPDATES}: "
            f"expected {expected_targets}, found {targets}"
        )
    if STAGE_UPDATES % RESTART_PERIOD_UPDATES:
        raise ValueError("continuation stage is not phase-matched to the LR restart period")

    root = Path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
    rows: list[dict[str, Any]] = []
    for stage, target in enumerate(targets, start=1):
        for source in ordered_sources:
            tag = str(source["dataset_tag"])
            steps_per_epoch = int(source["optimizer_steps_per_epoch"])
            source_config = _project_path(project_dir, source["config"])
            if not source_config.exists():
                raise FileNotFoundError(f"missing source config: {source_config}")

            run_name = _continuation_run_name(tag)
            checkpoint_dir = root / f"{run_name}_checkpoints"
            source_checkpoint = Path(source["expected_checkpoint"])
            source_epoch = int(source.get("expected_final_epoch", source_checkpoint.name.rsplit("-", 1)[-1]))
            seed_checkpoint = checkpoint_dir / f"checkpoint-epoch-{source_epoch:04d}"
            final_num_epochs = ceil_div(target, steps_per_epoch)
            final_epoch = final_num_epochs - 1
            expected_checkpoint = checkpoint_dir / f"checkpoint-epoch-{final_epoch:04d}"
            checkpoint_every_n_epochs = max(
                1, round(CHECKPOINT_EVERY_TARGET_UPDATES / steps_per_epoch)
            )
            config_path = (
                project_dir
                / "local"
                / CONTINUE_SWEEP_NAME
                / "configs"
                / f"stage{stage}_{run_name}.yaml"
            )

            if stage == 1:
                previous_checkpoint = seed_checkpoint
            else:
                previous_target = targets[stage - 2]
                previous_epoch = ceil_div(previous_target, steps_per_epoch) - 1
                previous_checkpoint = (
                    checkpoint_dir / f"checkpoint-epoch-{previous_epoch:04d}"
                )

            row = deepcopy(source)
            row.update(
                {
                    "manifest_version": MANIFEST_VERSION,
                    "sweep_name": CONTINUE_SWEEP_NAME,
                    "source_sweep_name": SOURCE_SWEEP_NAME,
                    "source_run_name": source["run_name"],
                    "source_config": _stored_path(project_dir, source_config),
                    "source_config_sha256": sha256_file(source_config),
                    "source_checkpoint": str(source_checkpoint),
                    "seed_checkpoint": str(seed_checkpoint),
                    "run_name": run_name,
                    "continue_stage": stage,
                    "fresh_initialization": False,
                    "full_state_resume": True,
                    "target_total_updates": target,
                    "stage_requested_updates": STAGE_UPDATES,
                    "restart_period_updates": RESTART_PERIOD_UPDATES,
                    "optimizer_steps_per_epoch": steps_per_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "final_num_epochs": final_num_epochs,
                    "expected_final_epoch": final_epoch,
                    "actual_total_updates": final_num_epochs * steps_per_epoch,
                    "checkpoint_every_target_updates": CHECKPOINT_EVERY_TARGET_UPDATES,
                    "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
                    "checkpoint_every_actual_updates": (
                        checkpoint_every_n_epochs * steps_per_epoch
                    ),
                    "checkpoint_dir": str(checkpoint_dir),
                    "previous_expected_checkpoint": str(previous_checkpoint),
                    "expected_checkpoint": str(expected_checkpoint),
                    "sample_label": f"dpm50_cont_{target // 1000}k",
                    "sample_path": (
                        f"results/{CONTINUE_SWEEP_NAME}/samples/"
                        f"{run_name}_seed123_dpm50_cont_{target // 1000}k.npz"
                    ),
                    "config": _stored_path(project_dir, config_path),
                    "note": (
                        "Full-state continuation from the frozen fresh300k-v2 run. "
                        "No legacy 200k or state-reset checkpoint is permitted."
                    ),
                }
            )
            rows.append(row)
    return rows


def build_continuation_config(
    source_path: Path, row: dict[str, Any]
) -> dict[str, Any]:
    config = yaml.safe_load(Path(source_path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"source config is not a mapping: {source_path}")
    result = deepcopy(config)
    result.setdefault("io", {})["output_dir"] = row["checkpoint_dir"]
    result.setdefault("train", {})["num_epochs"] = int(row["final_num_epochs"])
    result["train"]["checkpoint_every_n_epochs"] = int(
        row["checkpoint_every_n_epochs"]
    )
    return result


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], path))
        return flattened
    return {prefix: value}


def assert_continuation_config(
    source_path: Path,
    continuation_path: Path,
    row: dict[str, Any],
) -> None:
    source = yaml.safe_load(Path(source_path).read_text())
    continuation = yaml.safe_load(Path(continuation_path).read_text())
    expected = build_continuation_config(source_path, row)
    expected_flat = _flatten(expected)
    actual_flat = _flatten(continuation)
    differences = sorted(
        key
        for key in set(expected_flat) | set(actual_flat)
        if expected_flat.get(key) != actual_flat.get(key)
    )
    if differences:
        raise ValueError(
            f"{continuation_path} differs from the expected clone at: "
            + ", ".join(differences)
        )

    source_flat = _flatten(source)
    changed = sorted(
        key
        for key in set(source_flat) | set(actual_flat)
        if source_flat.get(key) != actual_flat.get(key)
    )
    forbidden = [key for key in changed if key not in ALLOWED_CONFIG_CHANGES]
    if forbidden:
        raise ValueError(
            "continuation config changed forbidden source keys: "
            + ", ".join(forbidden)
        )


def checkpoint_inventory(path: Path) -> dict[str, tuple[int, str]]:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"missing checkpoint directory: {path}")
    return {
        str(file.relative_to(path)): (file.stat().st_size, sha256_file(file))
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def copy_checkpoint_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)


def seed_continuation_directories(
    rows: list[dict[str, Any]],
    *,
    copy_function: Callable[[Path, Path], None] = copy_checkpoint_tree,
) -> None:
    first_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row["continue_stage"]) == 1:
            first_rows[str(row["dataset_tag"])] = row
    if set(first_rows) != set(EXPECTED_TAGS):
        raise ValueError("manifest does not contain one first-stage row per dataset tag")

    for tag in EXPECTED_TAGS:
        row = first_rows[tag]
        source = Path(row["source_checkpoint"])
        destination = Path(row["seed_checkpoint"])
        source_inventory = checkpoint_inventory(source)
        if not source_inventory:
            raise ValueError(f"source checkpoint is empty: {source}")
        if destination.exists():
            if checkpoint_inventory(destination) != source_inventory:
                raise ValueError(
                    f"existing seed checkpoint is not byte-identical: {destination}"
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_function(source, destination)
        if checkpoint_inventory(destination) != source_inventory:
            raise ValueError(f"copied checkpoint is not byte-identical: {destination}")


def build_analysis_manifest(
    source_rows: list[dict[str, Any]],
    continuation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = []
    for source in _validate_source_rows(source_rows):
        row = deepcopy(source)
        run_name = _continuation_run_name(str(source["dataset_tag"]))
        row.update(
            {
                "sweep_name": CONTINUE_SWEEP_NAME,
                "source_sweep_name": SOURCE_SWEEP_NAME,
                "source_run_name": source["run_name"],
                "source_checkpoint": source["expected_checkpoint"],
                "analysis_checkpoint": source["expected_checkpoint"],
                "analysis_updates": SOURCE_TARGET_UPDATES,
                "run_name": run_name,
                "sample_label": "dpm50_source_300k",
                "sample_path": (
                    f"results/{CONTINUE_SWEEP_NAME}/samples/"
                    f"{run_name}_seed123_dpm50_source_300k.npz"
                ),
            }
        )
        baseline.append(row)
    continuation = []
    for source in continuation_rows:
        row = deepcopy(source)
        row["analysis_checkpoint"] = source["expected_checkpoint"]
        row["analysis_updates"] = source["target_total_updates"]
        continuation.append(row)
    return baseline + continuation


def write_frozen_json(path: Path, value: Any, *, use_existing: bool) -> bool:
    path = Path(path)
    if path.exists() and use_existing:
        existing = json.loads(path.read_text())
        if existing != value:
            raise ValueError(f"{path} differs from the frozen content")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return True


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"manifest is not a list: {path}")
    return value


def _selected_rows(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_tags: list[str] | None,
    stages: list[int] | None,
) -> list[dict[str, Any]]:
    tag_filter = set(dataset_tags or EXPECTED_TAGS)
    stage_filter = set(stages or range(1, len(TARGET_UPDATES) + 1))
    return [
        row
        for row in rows
        if row["dataset_tag"] in tag_filter
        and int(row["continue_stage"]) in stage_filter
    ]


def _check_frozen_rows(project_dir: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(EXPECTED_TAGS) * len(TARGET_UPDATES):
        raise ValueError(f"continuation manifest must contain 50 rows; found {len(rows)}")
    for row in rows:
        source_path = _project_path(project_dir, row["source_config"])
        config_path = _project_path(project_dir, row["config"])
        if sha256_file(source_path) != row["source_config_sha256"]:
            raise ValueError(f"source config digest changed: {source_path}")
        assert_continuation_config(source_path, config_path, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--dataset-tag", action="append")
    parser.add_argument("--stage", type=int, action="append")
    parser.add_argument("--use-existing-manifest", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed-checkpoints", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    parser.add_argument("--print-field")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    source_path = args.source_manifest or source_manifest_path(project_dir)
    if not source_path.is_absolute():
        source_path = project_dir / source_path
    source_rows = _load_json_rows(source_path)
    proposed_rows = build_continuation_rows(
        project_dir,
        source_rows,
        checkpoint_root=args.checkpoint_root,
    )
    manifest_path = continuation_manifest_path(project_dir)

    if args.use_existing_manifest:
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing frozen continuation manifest: {manifest_path}")
        rows = _load_json_rows(manifest_path)
        if rows != proposed_rows:
            raise ValueError(
                "existing continuation manifest differs from the source-derived manifest"
            )
    else:
        rows = proposed_rows
        for row in rows:
            source_config = _project_path(project_dir, row["source_config"])
            destination = _project_path(project_dir, row["config"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            config = build_continuation_config(source_config, row)
            destination.write_text(yaml.safe_dump(config, sort_keys=False))
            assert_continuation_config(source_config, destination, row)
            print(f"Wrote {destination}")
        write_frozen_json(manifest_path, rows, use_existing=False)
        write_frozen_json(
            analysis_manifest_path(project_dir),
            build_analysis_manifest(source_rows, rows),
            use_existing=False,
        )
        print(f"Wrote {manifest_path}")
        print(f"Wrote {analysis_manifest_path(project_dir)}")

    _check_frozen_rows(project_dir, rows)
    selected = _selected_rows(
        rows, dataset_tags=args.dataset_tag, stages=args.stage
    )
    if args.seed_checkpoints:
        seed_continuation_directories(rows)
        print("Seeded ten isolated continuation directories from exact 300k checkpoints.")
    if args.print_field:
        if len(selected) != 1:
            raise SystemExit(
                f"--print-field requires exactly one selected row; found {len(selected)}"
            )
        print(selected[0][args.print_field])
    elif args.print_table:
        fields = (
            "continue_stage",
            "dataset_tag",
            "target_total_updates",
            "previous_expected_checkpoint",
            "expected_checkpoint",
        )
        print("\t".join(fields))
        for row in selected:
            print("\t".join(str(row[field]) for field in fields))
    elif args.check_only:
        print(f"Validated {len(rows)} frozen {CONTINUE_SWEEP_NAME} rows.")
    elif args.use_existing_manifest and not args.seed_checkpoints:
        print(f"Reused {len(rows)} frozen {CONTINUE_SWEEP_NAME} rows.")


if __name__ == "__main__":
    main()
