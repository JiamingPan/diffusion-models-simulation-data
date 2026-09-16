"""Paired, raw-weight DPM-50 trajectories from pure Gaussian noise.

Records the model's x0 estimate BEFORE each solver step, including patch
assignments and their persistence. The first persistence value is undefined.
High persistence alone is not proof of stitching: frozen noise and corrupted
single-map controls can also have stable nearest-patch assignments.

Initial noise is generated in CPU float32 and hashed per sample, then copied
unchanged to the requested device. This gives a new reproducible paired screen,
not the same sample indices as earlier GPU-generated sampling jobs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import sample_cosmodiff as sc  # noqa: E402
import patchwork_check as pw  # noqa: E402
from evaluate_late_start import boundary_ratio  # noqa: E402
from onestep_patchwork_probe import field_to_rgb  # noqa: E402

METRICS = ["majority", "persistence", "n_distinct", "boundary", "max_cos", "top_gap", "x0_rms", "patch_cos"]
COLORS = {"copy": (0, 114, 178), "junk": (213, 94, 0), "other": (100, 100, 100)}


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_initial_noise(n, shape, seed):
    if n <= 0:
        raise ValueError("n must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((n, *shape), dtype=torch.float32, generator=generator)


def noise_hashes(noise):
    return [array_hash(sample) for sample in noise.detach().cpu().numpy()]


def top2_cos(samples, reference):
    if len(reference) < 2:
        raise ValueError("top-2 cosine requires at least two reference maps")
    a = samples.reshape(len(samples), -1).astype(np.float32)
    b = reference.reshape(len(reference), -1).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    part = np.partition(a @ b.T, -2, axis=1)
    best = np.maximum(part[:, -1], part[:, -2])
    second = np.minimum(part[:, -1], part[:, -2])
    return best, second


def patch_matches(samples, normalized_reference):
    patches = pw.to_patches(samples)
    assign = np.empty(patches.shape[:2], dtype=np.int32)
    fit = np.empty(patches.shape[:2], dtype=np.float32)
    for p in range(patches.shape[1]):
        cos = patches[:, p] @ normalized_reference[:, p].T
        assign[:, p] = cos.argmax(axis=1)
        fit[:, p] = cos.max(axis=1)
    return assign, fit


@torch.no_grad()
def run(model, scheduler, reference, *, initial_noise, num_steps, batch_size,
        class_label, device, keep_steps):
    if batch_size <= 0 or num_steps <= 0:
        raise ValueError("batch_size and num_steps must be positive")
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError("trajectory probe requires v_prediction")
    if initial_noise.shape[1:] != reference.shape[1:]:
        raise ValueError("noise and reference shapes differ")
    scheduler.set_timesteps(num_steps)
    if len(scheduler.timesteps) != num_steps:
        raise ValueError("scheduler returned a different number of timesteps")
    n = len(initial_noise)
    x = initial_noise.to(device=device, dtype=torch.float32).clone()
    metrics = {m: np.full((num_steps, n), np.nan, dtype=np.float32) for m in METRICS}
    timesteps = np.asarray([int(t) for t in scheduler.timesteps], dtype=np.int64)
    rp = pw.to_patches(reference)
    assignments = np.empty((num_steps, n, rp.shape[1]), dtype=np.int32)
    kept = {}
    step_kwargs = {}
    if "generator" in inspect.signature(scheduler.step).parameters:
        step_kwargs["generator"] = torch.Generator(device=device).manual_seed(0)
    for k, t in enumerate(scheduler.timesteps):
        chunks = []
        for start in range(0, n, batch_size):
            inputs = x[start:start + batch_size]
            ts = torch.full((len(inputs),), int(t), device=device, dtype=torch.long)
            labels = torch.full_like(ts, class_label)
            chunks.append(model(inputs, timestep=ts, class_labels=labels, return_dict=False)[0])
        v = torch.cat(chunks)
        abar = scheduler.alphas_cumprod[int(t)].to(x)
        x0 = (abar.sqrt() * x - (1 - abar).sqrt() * v).float().cpu().numpy()
        if not np.isfinite(x0).all():
            raise ValueError(f"non-finite x0 estimate at step {k}, timestep {int(t)}")
        assign, fit = patch_matches(x0, rp)
        assignments[k] = assign
        majority, distinct, _ = pw.summarize(assign)
        metrics["majority"][k], metrics["n_distinct"][k] = majority, distinct
        if k:
            metrics["persistence"][k] = (assign == assignments[k - 1]).mean(axis=1)
        metrics["patch_cos"][k] = np.median(fit, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            metrics["boundary"][k] = boundary_ratio(x0)
        c1, c2 = top2_cos(x0, reference)
        metrics["max_cos"][k], metrics["top_gap"][k] = c1, c1 - c2
        metrics["x0_rms"][k] = np.sqrt((x0 ** 2).mean(axis=(1, 2, 3)))
        if k in keep_steps:
            kept[k] = {"x": x.float().cpu().numpy().copy(), "x0": x0.copy()}
        # One scheduler update for the whole batch, NOT one update per model chunk.
        x = scheduler.step(v, t, x, **step_kwargs).prev_sample
        if not torch.isfinite(x).all():
            raise ValueError(f"non-finite solver state at step {k}")
        if k % 10 == 0 or k == num_steps - 1:
            print(f"[trajectory] step={k} t={int(t)} majority_median={np.median(majority):.4f}", flush=True)
    return metrics, timesteps, kept, x.float().cpu().numpy(), assignments


def persistence_controls(reference, *, n, seed):
    rng = np.random.default_rng(seed)
    shape = (n, *reference.shape[1:])
    noise = rng.normal(size=shape).astype(np.float32)
    corrupted = pw.corrupted_single_maps(reference[np.arange(n) % len(reference)], cosine=.4, rng=rng)
    mosaic = pw.synthetic_patchworks(reference, count=n, rng=rng)
    rp = pw.to_patches(reference)
    cases = [
        ("independent_gaussian", noise, rng.normal(size=shape).astype(np.float32)),
        ("correlated_gaussian_rho098", noise, .98 * noise + np.sqrt(1 - .98**2) * rng.normal(size=shape)),
        ("frozen_gaussian", noise, noise),
        ("frozen_corrupted_single_map", corrupted, corrupted),
        ("frozen_synthetic_patchwork", mosaic, mosaic),
    ]
    rows = []
    for name, previous, current in cases:
        before, _ = patch_matches(previous, rp)
        after, fit = patch_matches(current, rp)
        majority, distinct, _ = pw.summarize(after)
        rows.append({"control": name, "n": n, "majority_median": float(np.median(majority)),
                     "persistence_median": float(np.median((before == after).mean(axis=1))),
                     "distinct_median": float(np.median(distinct)),
                     "patch_cos_median": float(np.median(fit)),
                     "max_cos_median": float(np.median(top2_cos(current, reference)[0]))})
    return rows


@contextmanager
def atomic_output(output):
    """Publish a complete per-process output directory; never overwrite old data."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    pending = Path(tempfile.mkdtemp(prefix=f".{output.name}.pending.", dir=output.parent))
    try:
        yield pending
        if output.exists():
            raise FileExistsError(f"output appeared during publication: {output}")
        os.rename(pending, output)
    except BaseException:
        # Retain interrupted artifacts for investigation, without a terminal PASS.
        print(f"INCOMPLETE OUTPUT RETAINED: {pending}", file=sys.stderr)
        raise
    finally:
        os.close(fd)
        lock.unlink()


def write_csv(path, rows):
    if not rows:
        raise ValueError("refusing to publish an empty result table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_completed_trajectory(path):
    report = json.loads(path.with_suffix(".json").read_text())
    if report.get("status") != "complete":
        raise ValueError(f"trajectory is not complete: {path}")
    hashes = report.get("artifacts_sha256", {})
    if path.name not in hashes or path.with_suffix(".csv").name not in hashes:
        raise ValueError(f"trajectory report lacks required artifact hashes: {path}")
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("invalid artifact name in trajectory manifest")
        actual = hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"trajectory artifact hash mismatch: {name}")


def write_figures(output, *, label, metrics, timesteps, snr, kept, final, final_cos, outcome):
    """Pillow-only charts: avoid the Great Lakes Matplotlib/libstdc++ failure."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    panel_w, panel_h = 350, 250
    canvas = Image.new("RGB", (4 * panel_w, 2 * panel_h + 60), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"{label}: model x0 estimates along DPM-50 (raw weights)", fill="black", font=font)
    draw.text((8, 25), "x: solver step, early -> late; gray band: SNR 0.1-0.5; faint lines: individual samples", fill="black", font=font)
    for index, metric in enumerate(METRICS):
        xleft = (index % 4) * panel_w + 55
        ytop = (index // 4) * panel_h + 85
        w, h = panel_w - 75, panel_h - 85
        values = metrics[metric]
        finite = values[np.isfinite(values)]
        upper = max(1e-6, float(finite.max()) * 1.05) if finite.size else 1
        lower = min(0., float(finite.min())) if finite.size else 0
        if metric in ("majority", "persistence", "patch_cos", "max_cos", "top_gap"):
            lower, upper = min(0., lower), max(1., upper)
        if metric == "n_distinct":
            upper = max(256., upper)
        def xy(step, value):
            return (xleft + step / max(1, len(timesteps) - 1) * w,
                    ytop + h - (value - lower) / (upper - lower) * h)
        region = np.flatnonzero((snr >= .1) & (snr <= .5))
        if len(region):
            draw.rectangle([xy(int(region[0]), upper), xy(int(region[-1]), lower)], fill=(242, 242, 242))
        draw.rectangle((xleft, ytop, xleft + w, ytop + h), outline=(160, 160, 160))
        draw.text((xleft, ytop - 22), metric, fill="black", font=font)
        for fraction in (0., .5, 1.):
            value = lower + fraction * (upper - lower)
            ypos = xy(0, value)[1]
            draw.line((xleft, ypos, xleft + w, ypos), fill=(220, 220, 220))
            draw.text((xleft - 50, ypos - 7), f"{value:.2f}", fill="black", font=font)
        for step in (0, (len(timesteps) - 1) // 2, len(timesteps) - 1):
            xpos = xy(step, lower)[0]
            draw.text((xpos - 5, ytop + h + 4), str(step), fill="black", font=font)
        for oc, color in COLORS.items():
            mask = outcome == oc
            if not mask.any():
                continue
            faint = tuple(int(.2 * channel + .8 * 255) for channel in color)
            for sample in np.flatnonzero(mask):
                points = [xy(k, float(values[k, sample])) for k in range(len(timesteps)) if np.isfinite(values[k, sample])]
                if len(points) > 1:
                    draw.line(points, fill=faint, width=1)
            median = np.asarray([np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in values[:, mask]])
            points = [xy(k, float(value)) for k, value in enumerate(median) if np.isfinite(value)]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
    legend_y = canvas.height - 20
    for j, (oc, color) in enumerate(COLORS.items()):
        draw.text((10 + j * 180, legend_y), f"{oc}: n={int((outcome == oc).sum())}", fill=color, font=font)
    canvas.save(output / f"{label}_trajectory.png")
    picks = [(oc, int(np.flatnonzero(outcome == oc)[0])) for oc in ("junk", "copy", "other") if (outcome == oc).any()]
    steps = sorted(kept)
    ih, iw = final.shape[-2:]
    cell_w, cell_h = max(180, iw), ih + 48
    gallery = Image.new("RGB", (cell_w * (len(steps) + 1), cell_h * len(picks) + 30), "white")
    gd = ImageDraw.Draw(gallery)
    gd.text((5, 5), f"{label}: x0 trajectory gallery (fixed display range -1 to 1)", fill="black", font=font)
    for row, (oc, sample) in enumerate(picks):
        for col, k in enumerate(steps):
            x, y = col * cell_w, row * cell_h + 30
            per = metrics["persistence"][k, sample]
            per_text = "NA" if not np.isfinite(per) else f"{per:.2f}"
            gd.text((x + 3, y), f"#{sample} {oc} step {k} t={timesteps[k]}", fill="black", font=font)
            gd.text((x + 3, y + 17), f"maj {metrics['majority'][k, sample]:.2f} per {per_text}", fill="black", font=font)
            gallery.paste(Image.fromarray(field_to_rgb(kept[k]["x0"][sample, 0])), (x, y + 40))
        x, y = len(steps) * cell_w, row * cell_h + 30
        gd.text((x + 3, y), f"Final #{sample} {oc}", fill="black", font=font)
        gd.text((x + 3, y + 17), f"max cos {final_cos[sample]:.4f}", fill="black", font=font)
        gallery.paste(Image.fromarray(field_to_rgb(final[sample, 0])), (x, y + 40))
    gallery.save(output / f"{label}_gallery.png")


def paired_outcomes(paths):
    loaded = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            loaded.append({key: data[key] for key in data.files})
    if len(loaded) < 2:
        raise ValueError("pairing requires at least two checkpoints")
    first = loaded[0]
    labels = [str(data["label"].item()) for data in loaded]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate model labels")
    for data in loaded:
        if not np.array_equal(data["initial_noise"], first["initial_noise"]):
            raise ValueError("initial noise differs across checkpoints")
        if not np.array_equal(data["initial_noise_sha256"], first["initial_noise_sha256"]):
            raise ValueError("initial noise hashes differ")
        if noise_hashes(torch.from_numpy(data["initial_noise"])) != data["initial_noise_sha256"].tolist():
            raise ValueError("recorded noise hashes do not match saved noise")
        for key in ("reference_sha256", "timesteps", "seed"):
            if not np.array_equal(data[key], first[key]):
                raise ValueError(f"{key} differs across checkpoints")
    rows = []
    for sample in range(len(first["final_cos"])):
        row = {"sample": sample, "initial_noise_sha256": str(first["initial_noise_sha256"][sample])}
        for label, data in zip(labels, loaded):
            row[f"{label}_outcome"] = str(data["outcome"][sample])
            row[f"{label}_final_cos"] = float(data["final_cos"][sample])
        rows.append(row)
    return rows


def paired_curve_rows(paths):
    """Use each L16 outcome mask on ALL models, preserving noise-by-noise pairs."""
    paired_outcomes(paths)
    loaded = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            loaded.append({key: data[key] for key in data.files})
    rows = []
    for anchor in loaded:
        anchor_label = str(anchor["label"].item())
        if "l16" not in anchor_label:
            continue
        for oc in COLORS:
            mask = anchor["outcome"] == oc
            if not mask.any():
                continue
            for target in loaded:
                for k, t in enumerate(target["timesteps"]):
                    row = {"cohort_model": anchor_label, "cohort_outcome": oc,
                           "label": str(target["label"].item()), "step": k,
                           "t": int(t), "snr": float(target["snr"][k]), "n": int(mask.sum())}
                    for metric in METRICS:
                        values = target[metric][k, mask]
                        row[metric] = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else ""
                    rows.append(row)
    if not rows:
        raise ValueError("paired cohort comparison needs an L16 checkpoint")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--checkpoint", type=Path)
    modes.add_argument("--pair-files", type=Path, nargs="+")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--keep-steps", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 40, 49])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    if args.pair_files:
        for path in args.pair_files:
            validate_completed_trajectory(path)
        rows = paired_outcomes(args.pair_files)
        curve_rows = paired_curve_rows(args.pair_files)
        with atomic_output(args.out_dir) as output:
            write_csv(output / "paired_outcomes.csv", rows)
            write_csv(output / "paired_by_l16_outcome.csv", curve_rows)
            all_rows = []
            for path in args.pair_files:
                with path.with_suffix(".csv").open() as handle:
                    all_rows.extend(csv.DictReader(handle))
            write_csv(output / "all_trajectories.csv", all_rows)
            (output / "complete.json").write_text(json.dumps({"status": "complete", "paired_samples": len(rows),
                "files": [str(path) for path in args.pair_files], "noise_reference_schedule_match": True}, indent=2) + "\n")
        print(f"PAIRED TRAJECTORIES PASSED: {len(rows)} identical initial-noise samples")
        return
    if args.config is None or args.label is None:
        parser.error("--checkpoint requires --config and --label")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("label must contain only letters, digits, underscores, and hyphens")
    if args.n <= 0 or args.batch_size <= 0 or args.num_steps != 50:
        parser.error("positive n/batch size and exactly 50 DPM steps are required for this screen")
    if any(k < 0 or k >= args.num_steps for k in args.keep_steps):
        parser.error("keep-steps outside the solver schedule")
    sc._reject_known_bad_runtime()
    sc._install_sklearn_roc_curve_stub()
    sc._ensure_cosmodiff_on_path(Path.cwd())
    checkpoint = args.checkpoint
    if not checkpoint.is_dir() or not sc._looks_like_checkpoint(checkpoint):
        raise FileNotFoundError(f"selected exact checkpoint is not loadable: {checkpoint}")
    model, scheduler = sc._load_for_sampling(checkpoint, args.config)
    scheduler = sc.build_inference_scheduler(scheduler, "DPMSolverMultistepScheduler")
    if str(scheduler.config.algorithm_type).startswith("sde"):
        raise ValueError("this screen requires deterministic ordinary DPM, not an SDE variant")
    device = torch.device(args.device)
    model.to(device).eval()
    reference = pw.load_reference(args.config, args.eval_root)
    if reference.shape != (256, 1, 128, 128):
        raise ValueError(f"expected 256 configured 128x128 training maps, got {reference.shape}")
    initial = make_initial_noise(args.n, reference.shape[1:], args.seed)
    hashes = noise_hashes(initial)
    print(f"INITIAL NOISE CPU-FLOAT32: batch_sha256={array_hash(initial.numpy())} n={args.n} seed={args.seed}", flush=True)
    metrics, timesteps, kept, final, assignments = run(
        model, scheduler, reference, initial_noise=initial, num_steps=args.num_steps,
        batch_size=args.batch_size, class_label=args.class_label, device=device, keep_steps=set(args.keep_steps))
    final_cos, _ = top2_cos(final, reference)
    outcome = np.where(final_cos > .98, "copy", np.where(final_cos < .8, "junk", "other"))
    snr = np.asarray([float(scheduler.alphas_cumprod[t] / (1 - scheduler.alphas_cumprod[t])) for t in timesteps])
    counts = {oc: int((outcome == oc).sum()) for oc in COLORS}
    rows = []
    for k in range(args.num_steps):
        for oc in COLORS:
            mask = outcome == oc
            if not mask.any():
                continue
            row = {"label": args.label, "step": k, "t": int(timesteps[k]), "snr": float(snr[k]), "outcome": oc, "n": int(mask.sum())}
            for metric in METRICS:
                values = metrics[metric][k, mask]
                row[metric] = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else ""
            rows.append(row)
    with atomic_output(args.out_dir) as output:
        prefix = output / f"{args.label}_trajectory"
        np.savez_compressed(prefix.with_suffix(".npz"), timesteps=timesteps, snr=snr,
            samples=final, final=final, final_cos=final_cos, outcome=outcome, label=args.label,
            seed=args.seed, initial_noise=initial.numpy(), initial_noise_sha256=np.asarray(hashes),
            reference_sha256=array_hash(reference), patch_assignments=assignments, **metrics,
            **{f"x0_step{k}": value["x0"] for k, value in kept.items()},
            **{f"x_step{k}": value["x"] for k, value in kept.items()})
        write_csv(prefix.with_suffix(".csv"), rows)
        controls = persistence_controls(reference, n=args.n, seed=args.seed)
        write_csv(output / "persistence_controls.csv", controls)
        write_figures(output, label=args.label, metrics=metrics, timesteps=timesteps, snr=snr,
                      kept=kept, final=final, final_cos=final_cos, outcome=outcome)
        provenance = {"status": "complete", "label": args.label, "checkpoint": str(checkpoint.resolve()),
            "config": str(args.config.resolve()), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "reference_sha256": array_hash(reference), "reference_count": len(reference), "raw_weights": True,
            "seed": args.seed, "n": args.n, "batch_size": args.batch_size, "class_label": args.class_label,
            "initial_noise_generator": "torch CPU float32", "initial_noise_sha256": hashes,
            "num_steps": args.num_steps, "timesteps": timesteps.tolist(), "scheduler_config": dict(scheduler.config),
            "outcome_counts": counts, "final_max_cos": final_cos.tolist(),
            "copy_threshold": .98, "junk_threshold": .8,
            "qualification": "Outcome groups are descriptive, not causal. High persistence alone does not prove stitching.",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "code_revision": os.environ.get("EXPECTED_COMMIT"),
            "runtime_code_root": os.environ.get("RUNTIME_CODE_ROOT"),
            "cosmodiff_pin_manifest": os.environ.get("COSMODIFF_PIN_MANIFEST"),
            "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}}
        prefix.with_suffix(".json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(f"{args.label}: outcomes {counts}", flush=True)
    print("per-sample final max_cos:", final_cos.tolist(), flush=True)
    for row in controls:
        print("[persistence-control]", json.dumps(row), flush=True)
    print(f"TRAJECTORY COMPLETE: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
