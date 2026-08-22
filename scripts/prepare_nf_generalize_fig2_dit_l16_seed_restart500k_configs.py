#!/usr/bin/env python
"""Freeze the two DiT-L16 seed restarts from exact 300k state to 500k."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


SOURCE_SWEEP_NAME = "nf_generalize_fig2_dit_l16_fresh300k_v2"
SWEEP_NAME = "nf_generalize_fig2_dit_l16_seed_restart500k_v1"
MANIFEST_VERSION = 1
RESUME_SEED = 456
SOURCE_TARGET_UPDATES = 300_000
TARGET_UPDATES = (340_000, 380_000, 420_000, 460_000, 500_000)
STAGE_UPDATES = 40_000
CHECKPOINT_EVERY_TARGET_UPDATES = 5_000
EXPECTED_TAGS = ("d2p08", "d2p10")
DEFAULT_CHECKPOINT_ROOT = Path(
    f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
)
ALLOWED_CONFIG_CHANGES = {
    "io.output_dir",
    "train.num_epochs",
    "train.checkpoint_every_n_epochs",
}


def ceil_div(numerator: int, denominator: int) -> int:
    if int(denominator) <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_dir) / path


def _stored_path(project_dir: Path, path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(project_dir).resolve()))
    except ValueError:
        return str(path)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], path))
        return flattened
    return {prefix: value}


def _validate_sources(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in source_rows if row.get("dataset_tag") in EXPECTED_TAGS]
    tags = [str(row.get("dataset_tag")) for row in selected]
    if sorted(tags) != sorted(EXPECTED_TAGS) or len(set(tags)) != len(EXPECTED_TAGS):
        raise ValueError(
            f"source manifest must contain exactly {EXPECTED_TAGS}; found {tags}"
        )
    for row in selected:
        tag = row["dataset_tag"]
        if row.get("sweep_name") != SOURCE_SWEEP_NAME:
            raise ValueError(f"{tag}: wrong source sweep")
        if row.get("arch") != "dit_l16":
            raise ValueError(f"{tag}: source is not DiT-L16")
        if int(row.get("target_total_updates", -1)) != SOURCE_TARGET_UPDATES:
            raise ValueError(f"{tag}: source target is not 300k")
        if int(row.get("optimizer_steps_per_epoch", 0)) <= 0:
            raise ValueError(f"{tag}: invalid optimizer_steps_per_epoch")
        for field in ("config", "expected_checkpoint", "expected_final_epoch"):
            if row.get(field) is None:
                raise ValueError(f"{tag}: missing source field {field}")
    return sorted(selected, key=lambda row: int(row["dataset_size"]))


def _run_name(tag: str) -> str:
    return f"nf_fig2_dit_l16_{tag}_noaug_seedrestart500k_v1_resume456"


def build_seed_restart_rows(
    project_dir: Path,
    source_rows: list[dict[str, Any]],
    *,
    checkpoint_root: Path | None = None,
) -> list[dict[str, Any]]:
    project_dir = Path(project_dir).resolve()
    sources = _validate_sources(source_rows)
    root = Path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
    rows: list[dict[str, Any]] = []
    for stage, target in enumerate(TARGET_UPDATES, start=1):
        for source in sources:
            tag = str(source["dataset_tag"])
            steps = int(source["optimizer_steps_per_epoch"])
            source_config = _project_path(project_dir, source["config"])
            if not source_config.is_file():
                raise FileNotFoundError(f"missing source config: {source_config}")
            run_name = _run_name(tag)
            checkpoint_dir = root / f"{run_name}_checkpoints"
            source_checkpoint = _project_path(
                project_dir, source["expected_checkpoint"]
            ).resolve()
            source_epoch = int(source["expected_final_epoch"])
            seed_checkpoint = checkpoint_dir / f"checkpoint-epoch-{source_epoch:04d}"
            final_num_epochs = ceil_div(target, steps)
            final_epoch = final_num_epochs - 1
            expected_checkpoint = checkpoint_dir / f"checkpoint-epoch-{final_epoch:04d}"
            checkpoint_every_epochs = max(
                1, round(CHECKPOINT_EVERY_TARGET_UPDATES / steps)
            )
            if stage == 1:
                previous_checkpoint = seed_checkpoint
            else:
                previous_epoch = ceil_div(TARGET_UPDATES[stage - 2], steps) - 1
                previous_checkpoint = checkpoint_dir / f"checkpoint-epoch-{previous_epoch:04d}"
            config_path = (
                project_dir
                / "local"
                / SWEEP_NAME
                / "configs"
                / f"stage{stage}_{run_name}.yaml"
            )
            row = deepcopy(source)
            row.update(
                {
                    "manifest_version": MANIFEST_VERSION,
                    "sweep_name": SWEEP_NAME,
                    "source_sweep_name": SOURCE_SWEEP_NAME,
                    "source_run_name": source["run_name"],
                    "source_config": _stored_path(project_dir, source_config),
                    "source_config_sha256": sha256_file(source_config),
                    "source_checkpoint": str(source_checkpoint),
                    "source_actual_total_updates": int(source["actual_total_updates"]),
                    "seed_checkpoint": str(seed_checkpoint),
                    "run_name": run_name,
                    "continue_stage": stage,
                    "fresh_initialization": False,
                    "full_state_resume": True,
                    "resume_seed": RESUME_SEED,
                    "apply_resume_seed": stage == 1,
                    "data_subset_seed": None,
                    "target_total_updates": int(target),
                    "stage_requested_updates": STAGE_UPDATES,
                    "optimizer_steps_per_epoch": steps,
                    "steps_per_epoch": steps,
                    "final_num_epochs": final_num_epochs,
                    "expected_final_epoch": final_epoch,
                    "actual_total_updates": final_num_epochs * steps,
                    "checkpoint_every_target_updates": CHECKPOINT_EVERY_TARGET_UPDATES,
                    "checkpoint_every_n_epochs": checkpoint_every_epochs,
                    "checkpoint_every_actual_updates": checkpoint_every_epochs * steps,
                    "checkpoint_dir": str(checkpoint_dir),
                    "previous_expected_checkpoint": str(previous_checkpoint),
                    "expected_checkpoint": str(expected_checkpoint),
                    "config": _stored_path(project_dir, config_path),
                    "audit_dir": _stored_path(
                        project_dir,
                        project_dir / "results" / SWEEP_NAME / "resume_audits",
                    ),
                }
            )
            rows.append(row)
    return rows


def build_seed_restart_config(source_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    source = yaml.safe_load(Path(source_path).read_text())
    if not isinstance(source, dict):
        raise ValueError(f"source config is not a mapping: {source_path}")
    result = deepcopy(source)
    result.setdefault("io", {})["output_dir"] = row["checkpoint_dir"]
    result.setdefault("train", {})["num_epochs"] = int(row["final_num_epochs"])
    result["train"]["checkpoint_every_n_epochs"] = int(
        row["checkpoint_every_n_epochs"]
    )
    return result


def assert_seed_restart_config(
    source_path: Path,
    continuation_path: Path,
    row: dict[str, Any],
) -> None:
    source = yaml.safe_load(Path(source_path).read_text())
    actual = yaml.safe_load(Path(continuation_path).read_text())
    expected = build_seed_restart_config(source_path, row)
    if actual != expected:
        differences = sorted(
            key
            for key in set(_flatten(actual)) | set(_flatten(expected))
            if _flatten(actual).get(key) != _flatten(expected).get(key)
        )
        raise ValueError("continuation config mismatch: " + ", ".join(differences))
    changed = sorted(
        key
        for key in set(_flatten(source)) | set(_flatten(actual))
        if _flatten(source).get(key) != _flatten(actual).get(key)
    )
    forbidden = [key for key in changed if key not in ALLOWED_CONFIG_CHANGES]
    if forbidden:
        raise ValueError("forbidden config changes: " + ", ".join(forbidden))


def checkpoint_inventory(path: Path) -> dict[str, tuple[int, str]]:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"missing checkpoint directory: {path}")
    return {
        str(file.relative_to(path)): (file.stat().st_size, sha256_file(file))
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def seed_restart_directories(rows: list[dict[str, Any]]) -> None:
    first_rows = {
        str(row["dataset_tag"]): row
        for row in rows
        if int(row["continue_stage"]) == 1
    }
    if set(first_rows) != set(EXPECTED_TAGS):
        raise ValueError("missing one first-stage row per seed-restart dataset")
    for tag in EXPECTED_TAGS:
        row = first_rows[tag]
        source = Path(row["source_checkpoint"])
        destination = Path(row["seed_checkpoint"])
        source_inventory = checkpoint_inventory(source)
        if not source_inventory:
            raise ValueError(f"source checkpoint is empty: {source}")
        if destination.exists():
            if checkpoint_inventory(destination) != source_inventory:
                raise ValueError(f"existing seed checkpoint is not byte-identical: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        if checkpoint_inventory(destination) != source_inventory:
            raise ValueError(f"copied checkpoint is not byte-identical: {destination}")


def _manifest_path(project_dir: Path) -> Path:
    return Path(project_dir) / "local" / SWEEP_NAME / "manifest.json"


def _source_manifest_path(project_dir: Path) -> Path:
    return Path(project_dir) / "local" / SOURCE_SWEEP_NAME / "manifest.json"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list):
        raise ValueError(f"manifest is not a list: {path}")
    return value


def selected_rows(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_tags: list[str] | None,
    stages: list[int] | None,
) -> list[dict[str, Any]]:
    tags = set(dataset_tags or EXPECTED_TAGS)
    stage_values = set(stages or range(1, len(TARGET_UPDATES) + 1))
    return [
        row
        for row in rows
        if row["dataset_tag"] in tags and int(row["continue_stage"]) in stage_values
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--dataset-tag", action="append")
    parser.add_argument("--stage", type=int, action="append")
    parser.add_argument("--use-existing-manifest", action="store_true")
    parser.add_argument("--seed-checkpoints", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    parser.add_argument("--print-field")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    source_manifest = args.source_manifest or _source_manifest_path(project_dir)
    if not source_manifest.is_absolute():
        source_manifest = project_dir / source_manifest
    sources = _load_rows(source_manifest)
    proposed = build_seed_restart_rows(
        project_dir,
        sources,
        checkpoint_root=args.checkpoint_root,
    )
    manifest_path = _manifest_path(project_dir)
    if args.use_existing_manifest:
        rows = _load_rows(manifest_path)
        if rows != proposed:
            raise ValueError("existing seed-restart manifest differs from proposed content")
    else:
        rows = proposed
        for row in rows:
            source_config = _project_path(project_dir, row["source_config"])
            config_path = _project_path(project_dir, row["config"])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(
                    build_seed_restart_config(source_config, row),
                    sort_keys=False,
                )
            )
            assert_seed_restart_config(source_config, config_path, row)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote {manifest_path}")

    for row in rows:
        assert_seed_restart_config(
            _project_path(project_dir, row["source_config"]),
            _project_path(project_dir, row["config"]),
            row,
        )
    if args.seed_checkpoints:
        seed_restart_directories(rows)
        print("Seeded two isolated restart directories with byte-identical 300k checkpoints.")

    chosen = selected_rows(
        rows,
        dataset_tags=args.dataset_tag,
        stages=args.stage,
    )
    if args.print_field:
        if len(chosen) != 1:
            raise SystemExit(f"--print-field requires one row; found {len(chosen)}")
        print(chosen[0][args.print_field])
    elif args.print_table:
        fields = (
            "continue_stage",
            "dataset_tag",
            "resume_seed",
            "target_total_updates",
            "previous_expected_checkpoint",
            "expected_checkpoint",
        )
        print("\t".join(fields))
        for row in chosen:
            print("\t".join(str(row[field]) for field in fields))
    elif args.check_only:
        print(f"Validated {len(rows)} frozen {SWEEP_NAME} rows.")


if __name__ == "__main__":
    main()
