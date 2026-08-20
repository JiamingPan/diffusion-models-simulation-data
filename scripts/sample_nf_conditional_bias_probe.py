#!/usr/bin/env python
"""Generate held-out continuous-cosmology samples for the HI bias probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SWEEP_NAME = "nf_conditional_bias_probe"


def parse_guidance_scale(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if cleaned in {"", "none", "null", "noguidance", "no_guidance"}:
        return None
    return float(cleaned)


def guidance_label(guidance_scale: float | None) -> str:
    if guidance_scale is None:
        return "noguidance"
    return f"g{float(guidance_scale):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def load_manifest(project_dir: Path, manifest_path: Path | None) -> list[dict[str, Any]]:
    path = manifest_path or project_dir / "local" / SWEEP_NAME / "manifest.json"
    with path.open() as f:
        return json.load(f)


def selected_rows(rows: list[dict[str, Any]], run_names: list[str] | None) -> list[dict[str, Any]]:
    if run_names:
        wanted = set(run_names)
        rows = [row for row in rows if row["run_name"] in wanted]
    return sorted(rows, key=lambda row: int(row["dataset_size"]))


def checkpoint_cli_args(row: dict[str, Any]) -> list[str]:
    """Pin sampling to a manifest checkpoint when one is explicitly requested."""

    checkpoint_epoch = row.get("checkpoint_epoch")
    if checkpoint_epoch is None:
        return []
    return ["--checkpoint_epoch", str(int(checkpoint_epoch))]


def output_path_for(project_dir: Path, row: dict[str, Any], seed: int, k: int, guidance_scale: float | None) -> Path:
    raw = str(row["sample_path"])
    label = guidance_label(guidance_scale)
    try:
        rel = raw.format(seed=seed, sample_label="dpm50", k=k, guidance=label)
    except KeyError:
        rel = raw.format(seed=seed, sample_label="dpm50", k=k)
    path = project_dir / rel
    if guidance_scale is not None and "{guidance}" not in raw:
        path = path.with_name(f"{path.stem}_{label}{path.suffix}")
    return path


def annotate_npz(path: Path, row: dict[str, Any], seed: int, k: int, project_dir: Path, guidance_scale: float | None) -> None:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    heldout_norm = np.load(row["heldout_sample_params_norm_path"]).astype(np.float32)
    heldout_indices = np.loadtxt(row["heldout_indices_path"], dtype=np.int64)
    heldout_raw_path = row.get("heldout_raw_params_path")
    if heldout_raw_path is None:
        heldout_raw_path = project_dir / "local" / SWEEP_NAME / "heldout" / "heldout_params_raw.npy"
    else:
        heldout_raw_path = Path(str(heldout_raw_path))
        if not heldout_raw_path.is_absolute():
            heldout_raw_path = project_dir / heldout_raw_path
    heldout_raw = np.load(heldout_raw_path).astype(np.float32)
    payload.update(
        {
            "run_name": np.array(row["run_name"]),
            "regime": np.array(row["regime"]),
            "dataset_size": np.array(int(row["dataset_size"])),
            "checkpoint_epoch": np.array(int(row.get("checkpoint_epoch", -1))),
            "requested_checkpoint": np.array(str(row.get("requested_checkpoint", ""))),
            "cfg_dropout": np.array(float(row.get("cfg_dropout", 0.0))),
            "guidance_scale": np.array(np.nan if guidance_scale is None else float(guidance_scale)),
            "guidance_label": np.array(guidance_label(guidance_scale)),
            "seed": np.array(int(seed)),
            "samples_per_cosmology": np.array(int(k)),
            "heldout_indices": heldout_indices,
            "theta_norm_repeated": heldout_norm,
            "theta_raw": heldout_raw,
        }
    )
    np.savez(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--cosmodiff-sample", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples-per-cosmology", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scheduler", default="DPMSolverMultistepScheduler")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--guidance-scale",
        action="append",
        help="Optional CFG guidance scale. May be repeated; use 'none' for no guidance.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    rows = selected_rows(load_manifest(project_dir, args.manifest), args.run_name)
    if not rows:
        raise SystemExit("No runs selected.")

    cosmodiff_dir = Path(os.environ.get("COSMODIFF_DIR", "/home/jiamingp/Diffusion_model/cosmo_diffusion_main"))
    cosmodiff_sample = Path(args.cosmodiff_sample) if args.cosmodiff_sample else cosmodiff_dir / "scripts" / "cosmodiff_sample.py"
    if not cosmodiff_sample.exists():
        raise FileNotFoundError(cosmodiff_sample)

    guidance_scales = [parse_guidance_scale(x) for x in args.guidance_scale] if args.guidance_scale else [None]

    for row in rows:
        config_path = project_dir / row["config"]
        with config_path.open() as f:
            config = yaml.safe_load(f)
        k = int(args.samples_per_cosmology or row.get("heldout_samples_per_cosmology", 64))
        n_samples = k * len(np.loadtxt(row["heldout_indices_path"], dtype=np.int64))
        for guidance_scale in guidance_scales:
            output_path = output_path_for(project_dir, row, args.seed, k, guidance_scale)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing {output_path}")
                continue

            cmd = [
                sys.executable,
                str(cosmodiff_sample),
                "--config",
                str(config_path),
                "--filepath",
                str(output_path),
                "--n_samples",
                str(n_samples),
                "--batch_size",
                str(args.batch_size),
                "--scheduler",
                str(args.scheduler),
                "--num_steps",
                str(args.num_steps),
                "--seed",
                str(args.seed),
                "--device",
                str(args.device),
                "--verbose",
            ]
            cmd.extend(checkpoint_cli_args(row))
            if guidance_scale is not None:
                cmd.extend(["--guidance_scale", str(float(guidance_scale))])
            print("Running:", " ".join(cmd), flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                raise SystemExit(rc)
            annotate_npz(output_path, row, args.seed, k, project_dir, guidance_scale)
            print(f"Wrote annotated sample file: {output_path}")


if __name__ == "__main__":
    main()
