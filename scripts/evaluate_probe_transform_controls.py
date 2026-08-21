#!/usr/bin/env python
"""Evaluate input-only transform controls with the existing frozen VGG probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simdiff_eval.probe_controls import (  # noqa: E402
    DEFAULT_K_CUTS,
    TransformSpec,
    aggregate_prediction_table,
    build_c0_specs,
    build_c1_specs,
    build_run_manifest,
    c0_symmetry_summary,
    c1_scale_cut_summary,
    evaluate_transform_specs,
    json_safe,
)
from simdiff_eval.probe_transforms import get_transform  # noqa: E402


EXPECTED_HELDOUT = np.arange(900, 932, dtype=np.int64)


def identity_specs() -> list[TransformSpec]:
    return [TransformSpec("identity", "identity", get_transform("identity"))]


def build_requested_specs(
    controls: list[str],
    *,
    roll_seed: int,
    k_cuts: list[float] | tuple[float, ...],
) -> tuple[list[TransformSpec], list[tuple[int, int]]]:
    requested = set(controls or ["identity"])
    unknown = requested.difference({"identity", "c0", "c1"})
    if unknown:
        raise ValueError(f"Unknown controls: {sorted(unknown)}")

    candidates: list[TransformSpec] = []
    roll_offsets: list[tuple[int, int]] = []
    if "c0" in requested:
        c0_specs, roll_offsets = build_c0_specs(roll_seed)
        candidates.extend(c0_specs)
    if "c1" in requested:
        candidates.extend(build_c1_specs(k_cuts))
    if not candidates:
        candidates.extend(identity_specs())

    specs_by_name: dict[str, TransformSpec] = {}
    for spec in candidates:
        specs_by_name.setdefault(spec.name, spec)
    specs = list(specs_by_name.values())
    if sum(spec.name == "identity" for spec in specs) != 1:
        raise ValueError("Combined controls must contain identity exactly once")
    return specs, roll_offsets


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _write_csv(path: Path, table: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def remove_unrequested_summaries(
    output_dir: Path,
    requested_controls: set[str],
) -> None:
    """Remove known summary artifacts that do not belong to this invocation."""
    summary_paths = {
        "c0": output_dir / "c0_symmetry_summary.json",
        "c1": output_dir / "c1_scale_cut_summary.json",
    }
    for control, path in summary_paths.items():
        if control not in requested_controls and path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run input-only controls with an existing frozen VGG cosmology probe."
    )
    parser.add_argument("--project-dir", default=".")
    parser.add_argument(
        "--source-project-dir",
        help="Checkout whose Git state is recorded in the output manifest; defaults to --project-dir.",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--encoder", default="results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--roll-seed", type=int, default=123)
    parser.add_argument(
        "--control",
        action="append",
        choices=("identity", "c0", "c1"),
        help="Control family to run. Repeat to combine families.",
    )
    parser.add_argument("--k-cut", action="append", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from evaluate_nf_conditional_bias_probe import load_vgg_encoder
    from prepare_nf_conditional_u128_config import DATA_ROOT
    from simdiff_eval.probe_eval import load_heldout_real_slices

    project_dir = Path(args.project_dir).resolve()
    source_project_dir = Path(args.source_project_dir or args.project_dir).resolve()
    data_root = args.data_root or DATA_ROOT
    encoder_path = Path(args.encoder)
    if not encoder_path.is_absolute():
        encoder_path = project_dir / encoder_path
    output_dir = args.output_dir or (
        project_dir / "results" / "nf_conditional_bias_probe" / "transform_controls"
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(encoder_path, allow_pickle=True) as data:
        normalization = data["normalization"].item()
        heldout = data["heldout_indices"].astype(np.int64)
    if not np.array_equal(heldout, EXPECTED_HELDOUT):
        raise ValueError(
            f"Encoder held-out indices must be 900..931; found {heldout.tolist()}"
        )

    images, theta_raw, sim_index, z_index = load_heldout_real_slices(
        data_root,
        heldout,
        slices_per_sim=128,
        norm=normalization,
    )
    encoder = load_vgg_encoder(project_dir, encoder_path, args.device)
    requested_controls = set(args.control or ["identity"])
    specs, roll_offsets = build_requested_specs(
        list(requested_controls),
        roll_seed=args.roll_seed,
        k_cuts=args.k_cut or DEFAULT_K_CUTS,
    )
    predictions = evaluate_transform_specs(
        images,
        theta_raw,
        sim_index,
        z_index,
        encoder,
        specs,
        batch_size=args.embedding_batch_size,
    )
    metrics = aggregate_prediction_table(
        predictions,
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    manifest = build_run_manifest(
        project_dir=source_project_dir,
        encoder_path=encoder_path,
        head_path=encoder.model_path,
        heldout_indices=heldout,
        slices_per_sim=128,
        transforms=[spec.manifest_record() for spec in specs],
        seeds={
            "bootstrap": int(args.bootstrap_seed),
            "roll": int(args.roll_seed),
        },
        arguments=vars(args),
        extra={"roll_offsets": roll_offsets} if roll_offsets else None,
    )

    predictions_path = output_dir / "probe_transform_predictions.csv"
    metrics_path = output_dir / "probe_transform_metrics.json"
    manifest_path = output_dir / "manifest.json"
    remove_unrequested_summaries(output_dir, requested_controls)
    _write_csv(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    if "c0" in requested_controls:
        c0_path = output_dir / "c0_symmetry_summary.json"
        c0_report = c0_symmetry_summary(
            predictions,
            n_boot=args.bootstrap,
            seed=args.bootstrap_seed,
        )
        _write_json(c0_path, c0_report)
        print(f"Wrote {c0_path}")
    if "c1" in requested_controls:
        c1_path = output_dir / "c1_scale_cut_summary.json"
        c1_report = c1_scale_cut_summary(
            predictions,
            n_boot=args.bootstrap,
            seed=args.bootstrap_seed,
        )
        _write_json(c1_path, c1_report)
        print(f"Wrote {c1_path}")
    _write_json(manifest_path, manifest)
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
