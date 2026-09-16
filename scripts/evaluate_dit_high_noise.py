#!/usr/bin/env python
"""Test whether DiT-L16 fails specifically in the terminal high-noise regime.

This is a checkpoint-only diagnostic.  It never trains or changes a checkpoint.
It writes one immutable NPZ plus a JSON summary for one model specification.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.dit_diagnostics import patch_boundary_per_image
from simdiff_eval.dit_high_noise import (
    batch_cosine,
    batch_nmse,
    choose_inference_begin,
    snr_and_v_weight,
    x0_from_v,
)
from simdiff_eval.io import load_real_reference_from_config
from simdiff_eval.torch_compat import install_torch_backend_compat

install_torch_backend_compat(entry_point=__name__)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_prediction(model, x_t, timesteps, labels):
    return model(
        x_t,
        timestep=timesteps,
        class_labels=labels,
        return_dict=False,
    )[0]


def _patch_ratio(images: np.ndarray, patch_size: int) -> np.ndarray:
    series = patch_boundary_per_image(images, patch_size=patch_size)
    return np.divide(
        series["boundary"],
        series["control"],
        out=np.full_like(series["boundary"], np.inf),
        where=series["control"] > 0.0,
    )


def _predict_x0_batches(model, x_t, timestep, alpha_bar, *, batch_size, device):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_t), batch_size):
            batch = x_t[start : start + batch_size].to(device)
            time = torch.full((len(batch),), int(timestep), device=device, dtype=torch.long)
            labels = torch.zeros(len(batch), device=device, dtype=torch.long)
            prediction = _model_prediction(model, batch, time, labels)
            outputs.append(x0_from_v(batch, prediction, alpha_bar).float().cpu())
    return torch.cat(outputs)


def _late_start_dpm(
    model,
    base_scheduler,
    source,
    noise,
    requested_timestep,
    *,
    num_steps,
    batch_size,
    device,
):
    from scripts.sample_cosmodiff import build_inference_scheduler

    scheduler = build_inference_scheduler(base_scheduler, "DPMSolverMultistepScheduler")
    set_timestep_parameters = inspect.signature(scheduler.set_timesteps).parameters
    set_timestep_kwargs = {"device": device} if "device" in set_timestep_parameters else {}
    scheduler.set_timesteps(int(num_steps), **set_timestep_kwargs)
    if not hasattr(scheduler, "set_begin_index"):
        raise RuntimeError(
            "The installed DPMSolverMultistepScheduler lacks set_begin_index; "
            "refusing an invalid hand-sliced img2img schedule."
        )
    begin_index, actual_timestep = choose_inference_begin(
        scheduler.timesteps, requested_timestep
    )
    scheduler.set_begin_index(begin_index)
    start_t = torch.full((len(source),), actual_timestep, dtype=torch.long)
    # Use the inference scheduler's own img2img path after setting begin_index.
    # This guarantees the starting alpha/sigma exactly matches the DPM schedule.
    current = scheduler.add_noise(source, noise, start_t).to(device)
    try:
        step_parameters = inspect.signature(scheduler.step).parameters
    except (TypeError, ValueError):
        step_parameters = {}
    generator = torch.Generator(device=device).manual_seed(7300 + actual_timestep)
    model.eval()
    with torch.inference_mode():
        for timestep in scheduler.timesteps[begin_index:]:
            pieces = []
            for start in range(0, len(current), batch_size):
                batch = current[start : start + batch_size]
                time = torch.full(
                    (len(batch),), int(timestep.item()), device=device, dtype=torch.long
                )
                labels = torch.zeros(len(batch), device=device, dtype=torch.long)
                pieces.append(_model_prediction(model, batch, time, labels))
            prediction = torch.cat(pieces)
            kwargs = {"generator": generator} if "generator" in step_parameters else {}
            current = scheduler.step(prediction, timestep, current, **kwargs).prev_sample
    return current.float().cpu(), begin_index, actual_timestep


def evaluate(args) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from scripts.sample_cosmodiff import _load_for_sampling

    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    if not checkpoint.is_dir() or not config.is_file():
        raise FileNotFoundError(f"missing checkpoint/config: {checkpoint}, {config}")
    device = torch.device(args.device)
    model, scheduler = _load_for_sampling(checkpoint, config)
    model = model.to(device)
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError("diagnostic requires a v_prediction checkpoint")
    patch_size = int(model.config.patch_size)
    if patch_size != 8:
        raise ValueError(f"expected patch_size=8, found {patch_size}")

    full_reference = np.asarray(
        load_real_reference_from_config(config, max_slices=None), dtype=np.float32
    )
    if len(full_reference) < args.num_reference:
        raise ValueError(
            f"requested {args.num_reference} references from only {len(full_reference)} maps"
        )
    selected_indices = np.linspace(
        0, len(full_reference) - 1, args.num_reference, dtype=np.int64
    )
    reference = np.asarray(full_reference[selected_indices], dtype=np.float32).copy()
    if reference.shape != (args.num_reference, 1, 128, 128):
        raise ValueError(f"unexpected reference shape {reference.shape}")
    source = torch.from_numpy(reference)
    noise_generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(source.shape, generator=noise_generator)
    alpha_bar = scheduler.alphas_cumprod.detach().cpu().float()
    snr, weight = snr_and_v_weight(alpha_bar, gamma=args.min_snr_gamma)

    requested_timesteps = np.asarray(args.timesteps, dtype=np.int64)
    if np.any(requested_timesteps < 0) or np.any(requested_timesteps >= len(alpha_bar)):
        raise ValueError("diagnostic timesteps fall outside the training schedule")
    x0_nmse, x0_cosine, x0_patch = [], [], []
    x0_gallery = []
    for timestep in requested_timesteps:
        time = torch.full((len(source),), int(timestep), dtype=torch.long)
        x_t = scheduler.add_noise(source, noise, time)
        predicted = _predict_x0_batches(
            model,
            x_t,
            int(timestep),
            alpha_bar[int(timestep)].to(device),
            batch_size=args.batch_size,
            device=device,
        ).numpy()
        x0_nmse.append(batch_nmse(predicted, reference))
        x0_cosine.append(batch_cosine(predicted, reference))
        x0_patch.append(_patch_ratio(predicted, patch_size))
        x0_gallery.append(predicted[: args.gallery_size])

    late_requested = np.asarray(args.late_start_timesteps, dtype=np.int64)
    late_outputs, late_actual, late_begin = [], [], []
    late_nmse, late_cosine, late_patch = [], [], []
    for timestep in late_requested:
        output, begin_index, actual_timestep = _late_start_dpm(
            model,
            scheduler,
            source,
            noise,
            int(timestep),
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            device=device,
        )
        array = output.numpy()
        late_outputs.append(array[: args.gallery_size])
        late_actual.append(actual_timestep)
        late_begin.append(begin_index)
        late_nmse.append(batch_nmse(array, reference))
        late_cosine.append(batch_cosine(array, reference))
        late_patch.append(_patch_ratio(array, patch_size))

    terminal_timestep = len(alpha_bar) - 1
    terminal_generator = torch.Generator().manual_seed(args.seed + 991)
    terminal_noise = torch.randn(source.shape, generator=terminal_generator)
    terminal_prediction = _predict_x0_batches(
        model,
        terminal_noise,
        terminal_timestep,
        alpha_bar[terminal_timestep].to(device),
        batch_size=args.batch_size,
        device=device,
    ).numpy()
    training_mean = full_reference.mean(axis=0, keepdims=True)
    repeated_mean = np.repeat(training_mean, len(reference), axis=0)

    arrays = {
        "reference": reference[: args.gallery_size],
        "reference_indices": selected_indices,
        "schedule_snr": snr,
        "schedule_v_weight": weight,
        "timesteps": requested_timesteps,
        "x0_nmse": np.stack(x0_nmse),
        "x0_cosine": np.stack(x0_cosine),
        "x0_patch_ratio": np.stack(x0_patch),
        "x0_gallery": np.stack(x0_gallery),
        "late_requested_timesteps": late_requested,
        "late_actual_timesteps": np.asarray(late_actual),
        "late_begin_indices": np.asarray(late_begin),
        "late_nmse": np.stack(late_nmse),
        "late_cosine": np.stack(late_cosine),
        "late_patch_ratio": np.stack(late_patch),
        "late_gallery": np.stack(late_outputs),
        "terminal_gallery": terminal_prediction[: args.gallery_size],
        "training_mean": training_mean,
        "terminal_nmse_to_training_mean": batch_nmse(terminal_prediction, repeated_mean),
        "terminal_patch_ratio": _patch_ratio(terminal_prediction, patch_size),
        "terminal_output_rms": np.sqrt(np.mean(terminal_prediction**2, axis=(1, 2, 3))),
        "reference_rms": np.sqrt(np.mean(reference**2, axis=(1, 2, 3))),
        "terminal_mean_mse": np.asarray(
            np.mean((terminal_prediction.mean(axis=0, keepdims=True) - training_mean) ** 2)
        ),
        "terminal_between_noise_rms": np.asarray(
            np.sqrt(np.mean((terminal_prediction - terminal_prediction.mean(axis=0)) ** 2))
        ),
        "training_between_map_rms": np.asarray(
            np.sqrt(np.mean((full_reference - training_mean) ** 2))
        ),
    }
    summary = {
        "status": "complete",
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "config_sha256": sha256(config),
        "code_revision": args.expected_code_revision,
        "prediction_type": scheduler.config.prediction_type,
        "num_train_timesteps": int(scheduler.config.num_train_timesteps),
        "patch_size": patch_size,
        "num_layers": int(model.config.num_layers),
        "num_reference": int(len(reference)),
        "complete_training_reference_size": int(len(full_reference)),
        "weights": "raw",
        "num_steps": int(args.num_steps),
        "sampling_seed": int(args.seed),
        "min_snr_gamma": float(args.min_snr_gamma),
        "terminal_timestep": terminal_timestep,
        "terminal_snr": float(snr[-1]),
        "terminal_v_weight": float(weight[-1]),
        "output_npz": str(args.output_npz.resolve()),
    }
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--num-reference", type=int, default=32)
    parser.add_argument("--gallery-size", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[499, 490, 475, 450, 425, 400, 350, 300, 200, 100, 25],
    )
    parser.add_argument(
        "--late-start-timesteps", type=int, nargs="+", default=[499, 450, 400, 300]
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") and args.device.startswith("cuda"):
        raise RuntimeError("CUDA diagnostic must run inside a Slurm allocation")
    args.output_npz = args.output_npz.resolve()
    summary_path = args.output_npz.with_suffix(".json")
    if args.output_npz.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite an existing diagnostic artifact")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    summary, arrays = evaluate(args)
    temporary_npz = args.output_npz.with_name(args.output_npz.name + ".tmp.npz")
    np.savez_compressed(temporary_npz, **arrays)
    os.replace(temporary_npz, args.output_npz)
    temporary_json = summary_path.with_name(summary_path.name + ".tmp")
    temporary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_json, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
