#!/usr/bin/env python
"""Resolve the exact d2p08 checkpoints used by the high-noise diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _old_model(project: Path, architecture: str, run_arch: str) -> dict:
    run = f"nf_fig2_dit_{run_arch}_d2p08_noaug_200k"
    config = project / "local/nf_generalize_fig2_dit/configs" / f"{run}.yaml"
    checkpoint = Path(
        "/scratch/huterer_root/huterer0/jiamingp/saved_runs/"
        f"nf_generalize_fig2_dit/{run}_checkpoints/checkpoint-epoch-6249"
    )
    return {"name": architecture, "checkpoint": str(checkpoint), "config": str(config)}


def resolve(project: Path) -> list[dict]:
    rows = [
        _old_model(project, "dit_l8_200k", "l8"),
        _old_model(project, "dit_l12_200k", "base"),
    ]
    manifest_path = project / "local/nf_generalize_fig2_dit_l16_seed_restart500k_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = [row for row in manifest if row["dataset_tag"] == "d2p08"]
    stage1 = next(row for row in selected if int(row["continue_stage"]) == 1)
    stage5 = next(row for row in selected if int(row["continue_stage"]) == 5)
    source_config = Path(stage1["source_config"])
    final_config = Path(stage5["config"])
    rows.extend(
        [
            {
                "name": "dit_l16_fresh300k",
                "checkpoint": stage1["source_checkpoint"],
                "config": str(source_config if source_config.is_absolute() else project / source_config),
            },
            {
                "name": "dit_l16_seed456_500k",
                "checkpoint": stage5["expected_checkpoint"],
                "config": str(final_config if final_config.is_absolute() else project / final_config),
            },
        ]
    )
    expected_layers = [8, 12, 16, 16]
    configs = []
    for row, depth in zip(rows, expected_layers):
        checkpoint, config = Path(row["checkpoint"]), Path(row["config"])
        if not checkpoint.is_dir() or not config.is_file():
            raise FileNotFoundError(f"missing {row['name']} input: {checkpoint}, {config}")
        payload = yaml.safe_load(config.read_text())
        configs.append(payload)
        found = int(payload["model"]["kwargs"]["num_layers"])
        if found != depth:
            raise ValueError(f"{row['name']} expected {depth} layers, found {found}")
        if int(payload["model"]["kwargs"]["patch_size"]) != 8:
            raise ValueError(f"{row['name']} is not patch_size=8")
        if payload["noise_scheduler"]["kwargs"]["prediction_type"] != "v_prediction":
            raise ValueError(f"{row['name']} is not v_prediction")
        if float(payload["train"]["min_snr_gamma"]) != 5.0:
            raise ValueError(f"{row['name']} does not use min_snr_gamma=5")
    reference_data = json.dumps(configs[0]["data"], sort_keys=True)
    reference_noise = json.dumps(configs[0]["noise_scheduler"], sort_keys=True)
    for row, payload in zip(rows[1:], configs[1:]):
        if json.dumps(payload["data"], sort_keys=True) != reference_data:
            raise ValueError(f"{row['name']} data config differs from L8")
        if json.dumps(payload["noise_scheduler"], sort_keys=True) != reference_noise:
            raise ValueError(f"{row['name']} noise schedule differs from L8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    rows = resolve(args.project_dir.resolve())
    payload = rows if args.index is None else rows[args.index]
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
