"""One forward pass: does the DiT's x0 estimate stitch patches from different training maps?

For each timestep t in --timesteps and each of two input types
  no_trace : x_t = sqrt(abar) * training_mean + sqrt(1-abar) * eps   (what a generative trajectory sees)
  trace    : x_t = sqrt(abar) * real_map_i    + sqrt(1-abar) * eps   (what training saw)
the frozen model predicts v once, x0_hat = sqrt(abar) x_t - sqrt(1-abar) v, and x0_hat is scored with
  majority_fraction : fraction of 8x8 patches whose nearest same-location training patch is the modal map
  n_distinct        : number of distinct training maps the patches point to
  boundary_ratio    : |jump| at 8-px patch boundaries / elsewhere
  max_cos           : whole-image max cosine to the training set
If L16's no_trace x0_hat is a patchwork (low majority, many maps, boundary >> 1) while L8's is coherent,
the failure is the network's own function at mid-noise, not solver accumulation.

    python scripts/onestep_patchwork_probe.py --checkpoint <ckpt> --config <run.yaml> --label dit_l16_fresh300k \
        --eval-root <simdiff_eval checkout> --timesteps 449 399 349 299 249 --n 64 --out-dir results/onestep_probe
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
import patchwork_check as pw  # noqa: E402
from evaluate_late_start import boundary_ratio  # noqa: E402


@torch.no_grad()
def predict_x0(model, scheduler, x_t: torch.Tensor, t: int, class_label: int) -> torch.Tensor:
    abar = scheduler.alphas_cumprod[t].to(x_t)
    ts = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
    labels = torch.full((x_t.shape[0],), class_label, device=x_t.device, dtype=torch.long)
    v = model(x_t, timestep=ts, class_labels=labels, return_dict=False)[0]
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError(scheduler.config.prediction_type)
    return abar.sqrt() * x_t - (1 - abar).sqrt() * v


def add_shared_noise_pair(
    scheduler,
    *,
    mean: torch.Tensor,
    traced: torch.Tensor,
    noise: torch.Tensor,
    timestep: int,
) -> dict[str, torch.Tensor]:
    """Noise both sources with exactly the same epsilon realization."""
    if mean.shape != traced.shape or mean.shape != noise.shape:
        raise ValueError(
            f"mean, trace, and noise shapes must match: "
            f"{mean.shape}, {traced.shape}, {noise.shape}"
        )
    ts = torch.full(
        (mean.shape[0],), timestep, device=mean.device, dtype=torch.long
    )
    return {
        "no_trace": scheduler.add_noise(mean, noise, ts),
        "trace": scheduler.add_noise(traced, noise, ts),
    }


def field_to_rgb(field: np.ndarray) -> np.ndarray:
    """Small dependency-free approximation to viridis for values in [-1, 1]."""
    anchors = np.asarray(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    scaled = np.clip((field.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0)
    position = scaled * (len(anchors) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (position - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    return np.rint(rgb).astype(np.uint8)


def write_gallery_png(
    gallery: dict[tuple[int, str], np.ndarray],
    *,
    timesteps: list[int],
    label: str,
    output: Path,
) -> None:
    """Write the diagnostic gallery without importing system Matplotlib."""
    from PIL import Image, ImageDraw

    sample = next(iter(gallery.values()))
    height, width = sample.shape[-2:]
    label_width = 110
    title_height = 28
    rows = 2 * len(timesteps)
    canvas = Image.new(
        "RGB", (label_width + 4 * width, title_height + rows * height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 7), f"{label}: one-step x0 estimates", fill="black")
    for row_index, timestep in enumerate(timesteps):
        for kind_offset, kind in enumerate(("no_trace", "trace")):
            row = 2 * row_index + kind_offset
            y = title_height + row * height
            draw.text((5, y + height // 2 - 6), f"t={timestep} {kind}", fill="black")
            for column in range(4):
                rgb = field_to_rgb(gallery[(timestep, kind)][column, 0])
                canvas.paste(Image.fromarray(rgb, mode="RGB"), (label_width + column * width, y))
    canvas.save(output, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[449, 399, 349, 299, 249])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.n <= 0 or args.batch_size <= 0:
        raise ValueError("--n and --batch-size must be positive")
    if not args.timesteps or len(set(args.timesteps)) != len(args.timesteps):
        raise ValueError("--timesteps must be a non-empty list without duplicates")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.out_dir / f"{args.label}_onestep.csv"
    output_png = args.out_dir / f"{args.label}_onestep.png"
    output_json = args.out_dir / f"{args.label}_onestep.json"
    for output in (output_csv, output_png, output_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    sc._reject_known_bad_runtime(); sc._install_sklearn_roc_curve_stub(); sc._ensure_cosmodiff_on_path(Path.cwd())
    requested_checkpoint = Path(args.checkpoint)
    checkpoint = requested_checkpoint
    if checkpoint.is_dir() and not sc._looks_like_checkpoint(checkpoint):
        latest = sc._find_latest_checkpoint(checkpoint)
        if latest is None:
            raise FileNotFoundError(f"no loadable checkpoint under {checkpoint}")
        checkpoint = latest
    model, scheduler = sc._load_for_sampling(checkpoint, args.config)  # training DDPM scheduler, raw weights
    device = torch.device(args.device)
    model.to(device).eval()

    if getattr(scheduler.config, "prediction_type", None) != "v_prediction":
        raise ValueError("one-step probe requires a v_prediction checkpoint")
    train_timesteps = len(scheduler.alphas_cumprod)
    if min(args.timesteps) < 0 or max(args.timesteps) >= train_timesteps:
        raise ValueError(
            f"requested timesteps {args.timesteps} outside [0, {train_timesteps - 1}]"
        )

    reference = pw.load_reference(args.config, args.eval_root)
    mean = reference.mean(axis=0, keepdims=True)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    src_idx = rng.integers(0, len(reference), size=args.n)

    rows, gallery = [], {}
    for t in args.timesteps:
        mean_source = np.repeat(mean, args.n, axis=0)
        trace_source = reference[src_idx]
        noise = torch.randn(
            (args.n, *mean.shape[1:]), device=device, generator=gen
        )
        for kind in ("no_trace", "trace"):
            preds = []
            for start in range(0, args.n, args.batch_size):
                end = start + args.batch_size
                paired = add_shared_noise_pair(
                    scheduler,
                    mean=torch.from_numpy(mean_source[start:end]).to(device),
                    traced=torch.from_numpy(trace_source[start:end]).to(device),
                    noise=noise[start:end],
                    timestep=t,
                )
                x_t = paired[kind]
                preds.append(predict_x0(model, scheduler, x_t, t, args.class_label).float().cpu().numpy())
            x0_hat = np.concatenate(preds)
            majority, distinct, _ = pw.summarize(pw.patch_assignments(x0_hat, reference))
            br = boundary_ratio(x0_hat)
            mc = pw.whole_image_max_cos(x0_hat, reference)
            snr = float(scheduler.alphas_cumprod[t] / (1 - scheduler.alphas_cumprod[t]))
            row = {"label": args.label, "t": t, "snr": snr, "input": kind, "n": args.n,
                   "majority_median": float(np.median(majority)), "distinct_median": float(np.median(distinct)),
                   "boundary_median": float(np.median(br)), "max_cos_median": float(np.median(mc)),
                   "x0hat_rms": float(np.sqrt((x0_hat ** 2).mean()))}
            rows.append(row)
            gallery[(t, kind)] = x0_hat[:4]
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    token = f"{os.getpid()}"
    temporary_csv = args.out_dir / f".{args.label}.{token}.tmp.csv"
    temporary_png = args.out_dir / f".{args.label}.{token}.tmp.png"
    temporary_json = args.out_dir / f".{args.label}.{token}.tmp.json"
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

    write_gallery_png(
        gallery,
        timesteps=list(args.timesteps),
        label=args.label,
        output=temporary_png,
    )
    provenance = {
        "status": "complete",
        "label": args.label,
        "requested_checkpoint": str(requested_checkpoint),
        "resolved_checkpoint": str(checkpoint),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "training_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
        "reference_count": int(len(reference)),
        "timesteps": list(args.timesteps),
        "n": int(args.n),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "class_label": int(args.class_label),
        "weights": "raw",
        "paired_noise": True,
    }
    temporary_json.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    try:
        os.replace(temporary_png, output_png)
        os.replace(temporary_csv, output_csv)
        os.replace(temporary_json, output_json)
    finally:
        for temporary in (temporary_csv, temporary_png, temporary_json):
            temporary.unlink(missing_ok=True)
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
