"""Late-start DPM sampling: skip the high-noise window that Min-SNR left untrained.

Mechanism test for the DiT-L16 blocky-sample failure. Instead of starting the
DPM-Solver trajectory from pure noise at t = T-1, start at a chosen training
timestep t_start from

    x_{t_start} = sqrt(abar_t) * x_bar + sqrt(1 - abar_t) * eps,

where x_bar is the mean of the exact configured training subset (which is also
what every model predicts at the terminal timestep) and eps ~ N(0, I).  The
remaining DPM-50 steps (all schedule timesteps <= t_start) run unchanged, with
the solver's own step-index lookup so sigmas stay aligned with the full 50-step
schedule.

Everything else (checkpoint loading, scheduler construction, post-hoc EMA,
provenance fields) is delegated to scripts/sample_cosmodiff.py so results are
directly comparable with the existing DPM-50 sample files.

Run from the code root, inside a Slurm allocation, e.g.

    python scripts/sample_late_start.py \
        --checkpoint <ckpt-dir> --config <run.yaml> \
        --t-start 399 --num-samples 512 --batch-size 8 --seed 123 \
        --output results/late_start/d2p08_l16_fresh300k_t399.npz

--t-start 499 (or larger than the top schedule timestep) reproduces the
ordinary full-schedule DPM-50 run and is the control.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sample_cosmodiff as sc  # noqa: E402


def load_training_mean(config_path: Path, eval_root: Path | None, npy: Path | None) -> tuple[np.ndarray, dict]:
    """Mean of the exact configured training subset, shape (1, 1, H, W)."""
    if npy is not None:
        arr = np.load(npy)
        arr = np.asarray(arr, dtype=np.float32).reshape(1, 1, *arr.shape[-2:])
        return arr, {"training_mean_source": str(npy)}
    roots = [eval_root] if eval_root is not None else []
    roots += [Path.cwd(), Path.cwd() / "scripts"]
    for root in roots:
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from simdiff_eval.io import iter_real_reference_batches_from_config
    except ImportError as exc:
        raise SystemExit(
            "simdiff_eval is not importable; pass --eval-root <checkout containing simdiff_eval> "
            f"or --training-mean-npy. ({exc})"
        )
    batches = list(iter_real_reference_batches_from_config(config_path))
    real = np.concatenate(batches).astype(np.float32)
    if real.ndim != 4:
        raise ValueError(f"unexpected reference shape {real.shape}")
    if not np.isfinite(real).all():
        raise ValueError("non-finite values in training reference")
    mean = real.mean(axis=0, keepdims=True)
    return mean, {"training_mean_source": "config", "training_reference_count": int(real.shape[0])}


def pick_start_timestep(scheduler, requested: int) -> tuple[int, int]:
    """Largest schedule timestep <= requested; returns (timestep, index into scheduler.timesteps)."""
    ts = [int(t) for t in scheduler.timesteps]
    candidates = [(t, i) for i, t in enumerate(ts) if t <= requested]
    if not candidates:
        raise ValueError(f"no schedule timestep <= {requested}; schedule top is {ts[0]}")
    return max(candidates)


@torch.no_grad()
def generate_late_start(model, scheduler, *, x_bar: torch.Tensor, t_start: int, init: str,
                        batch_size: int, num_steps: int, device: torch.device,
                        generator: torch.Generator, class_labels: torch.Tensor | None):
    model.eval()
    set_timestep_parameters = inspect.signature(scheduler.set_timesteps).parameters
    set_timestep_kwargs = {"device": device} if "device" in set_timestep_parameters else {}
    scheduler.set_timesteps(num_steps, **set_timestep_kwargs)
    t_actual, start_idx = pick_start_timestep(scheduler, t_start)
    if not hasattr(scheduler, "set_begin_index"):
        raise RuntimeError(
            "The inference scheduler lacks set_begin_index; refusing a late-start "
            "trajectory whose sigma state cannot be aligned."
        )
    scheduler.set_begin_index(start_idx)
    shape = (batch_size, *x_bar.shape[1:])
    eps = torch.randn(shape, device=device, generator=generator)
    t_tensor = torch.full((batch_size,), t_actual, device=device, dtype=torch.long)
    if init == "mean":
        x0 = x_bar.expand(shape)
    elif init == "zero":
        x0 = torch.zeros(shape, device=device)
    else:
        raise ValueError(init)
    images = scheduler.add_noise(x0, eps, t_tensor)

    try:
        step_params = inspect.signature(scheduler.step).parameters
    except (TypeError, ValueError):
        step_params = {}
    step_kwargs = {"generator": generator} if "generator" in step_params else {}

    for t in scheduler.timesteps[start_idx:]:
        timesteps = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
        if class_labels is not None:
            pred = model(images, timestep=timesteps,
                         class_labels=class_labels.to(device=device, dtype=torch.long),
                         return_dict=False)[0]
        else:
            pred = model(images, timesteps, return_dict=False)[0]
        images = scheduler.step(pred, t, images, **step_kwargs).prev_sample
    return images, t_actual, int(len(scheduler.timesteps) - start_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Run YAML (scheduler + data subset).")
    parser.add_argument("--output", required=True, help="Output .npz path (refuses to overwrite).")
    parser.add_argument("--t-start", type=int, required=True,
                        help="Training timestep to start from; the largest schedule timestep <= this is used.")
    parser.add_argument("--init", choices=["mean", "zero"], default="mean",
                        help="x0 used to build x_{t_start}: training mean (default) or zero.")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scheduler", default="DPMSolverMultistepScheduler")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--ema-sigma-rel", type=float, default=None,
                        help="Post-hoc EMA profile, as in sample_cosmodiff.py. Omit for raw weights.")
    parser.add_argument("--eval-root", type=Path, default=None,
                        help="Checkout containing simdiff_eval (for the training mean).")
    parser.add_argument("--training-mean-npy", type=Path, default=None,
                        help="Precomputed training-mean array instead of loading the subset.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    if output.suffix != ".npz":
        raise ValueError("--output must be .npz so provenance can be stored")

    sc._reject_known_bad_runtime()
    sc._install_sklearn_roc_curve_stub()
    sc._ensure_cosmodiff_on_path(Path.cwd())

    requested_checkpoint = Path(args.checkpoint)
    checkpoint = requested_checkpoint
    if checkpoint.is_dir() and not sc._looks_like_checkpoint(checkpoint):
        latest = sc._find_latest_checkpoint(checkpoint)
        if latest is None:
            raise FileNotFoundError(f"no checkpoint under {checkpoint}")
        checkpoint = latest
    config_path = Path(args.config)

    model, scheduler = sc._load_for_sampling(checkpoint, config_path)
    model = sc._apply_posthoc_ema(model, requested_checkpoint=requested_checkpoint,
                                  resolved_checkpoint=checkpoint, sigma_rel=args.ema_sigma_rel)
    scheduler = sc.build_inference_scheduler(scheduler, args.scheduler)
    if getattr(scheduler.config, "prediction_type", None) != "v_prediction":
        raise ValueError("late-start mechanism test requires a v_prediction checkpoint")
    scheduler.set_timesteps(args.num_steps)
    audit = sc.scheduler_audit_metadata(scheduler, args.num_steps)

    mean_np, mean_meta = load_training_mean(config_path, args.eval_root, args.training_mean_npy)
    device = torch.device(args.device)
    model.to(device).eval()
    x_bar = torch.from_numpy(mean_np).to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    labels = torch.full((args.batch_size,), args.class_label, dtype=torch.long)
    batches, remaining = [], args.num_samples
    t_actual = steps_run = None
    while remaining > 0:
        n = min(args.batch_size, remaining)
        imgs, t_actual, steps_run = generate_late_start(
            model, scheduler, x_bar=x_bar, t_start=args.t_start, init=args.init,
            batch_size=n, num_steps=args.num_steps, device=device, generator=generator,
            class_labels=labels[:n])
        if t_actual is None or steps_run is None:
            raise RuntimeError("late-start scheduler did not report its executed range")
        batches.append(imgs.detach().cpu().numpy())
        remaining -= n
    samples = np.concatenate(batches, axis=0)
    if not np.isfinite(samples).all():
        raise RuntimeError("non-finite samples produced")

    extra = dict(audit)
    extra.update({
        "late_start_requested": int(args.t_start),
        "late_start_actual": int(t_actual),
        "late_start_steps_run": int(steps_run),
        "late_start_init": str(args.init),
        "ema_sigma_rel": float(args.ema_sigma_rel) if args.ema_sigma_rel is not None else -1.0,
        "training_mean": mean_np,
        "training_mean_sha256": hashlib.sha256(mean_np.tobytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        **mean_meta,
    })
    temporary = output.with_name(output.name + ".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    sc.save_sample_output(temporary, samples, requested_checkpoint=requested_checkpoint,
                          resolved_checkpoint=checkpoint, config_path=config_path,
                          scheduler_name=scheduler.__class__.__name__, num_steps=args.num_steps,
                          seed=args.seed, scheduler_audit=extra)
    os.replace(temporary, output)
    summary = {k: (v if not isinstance(v, np.ndarray) else f"array{v.shape}") for k, v in extra.items()}
    print(json.dumps({"output": str(output), "samples": list(samples.shape), **summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
