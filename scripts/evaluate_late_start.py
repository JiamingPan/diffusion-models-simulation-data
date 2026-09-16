"""Score late-start (and ordinary) DPM sample files against the exact training subset.

For every .npz produced by sample_late_start.py or sample_cosmodiff.py it reports:

  copy_fraction      fraction of samples whose max pixel-cosine to any training map > 0.98
  junk_fraction      fraction with max cosine < 0.8  (the blocky / speckle population)
  median_max_cos     median over samples of the max cosine
  boundary_ratio     mean |pixel jump| across 8-px patch boundaries / mean jump elsewhere
  boundary_copies    same, restricted to the copy population
  boundary_noncopies same, restricted to everything else
  pk_ratio_hi        mean generated / training power averaged over 32 <= k <= 64
  pk_ratio_max_dev   max |ratio - 1| over k <= 64 (axial Nyquist for 128^2 maps)

The decision for the mechanism test is: at t_start ~ 399, L16's junk_fraction and
boundary_noncopies should collapse toward 0 and 1.0 while L8/L12 stay where they
were at t_start = 499.

Usage (from the code root or with --eval-root pointing at the simdiff_eval checkout):

    python scripts/evaluate_late_start.py --config <run.yaml> \
        --samples results/late_start/*.npz --out results/late_start/summary.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

COPY_THRESHOLD = 0.98
JUNK_THRESHOLD = 0.80
PATCH = 8
NYQUIST = 64


def scalar(data, key: str):
    if key not in data:
        raise ValueError(f"sample file lacks required provenance field {key!r}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"sample provenance {key!r} is not scalar: {value.shape}")
    return value.reshape(-1)[0].item()


def validate_provenance(data, *, training_mean_sha256: str) -> dict:
    record = {
        "t_start_requested": int(scalar(data, "late_start_requested")),
        "t_start": int(scalar(data, "late_start_actual")),
        "steps_run": int(scalar(data, "late_start_steps_run")),
        "seed": int(scalar(data, "seed")),
        "num_steps": int(scalar(data, "num_steps")),
        "ema_sigma_rel": float(scalar(data, "ema_sigma_rel")),
        "checkpoint": str(scalar(data, "resolved_checkpoint")),
        "config_sha256": str(scalar(data, "config_sha256")),
    }
    if str(scalar(data, "training_mean_sha256")) != training_mean_sha256:
        raise ValueError("sample training subset does not match the evaluation reference")
    if str(scalar(data, "late_start_init")) != "mean":
        raise ValueError("sample was not initialized from the configured training mean")
    if str(scalar(data, "scheduler_class")) != "DPMSolverMultistepScheduler":
        raise ValueError("sample does not use DPMSolverMultistepScheduler")
    if record["num_steps"] != 50 or record["seed"] != 123:
        raise ValueError("sample does not use the frozen 50-step, seed-123 protocol")
    if record["ema_sigma_rel"] != -1.0:
        raise ValueError("sample does not use raw weights")
    if not 0 <= record["t_start"] <= record["t_start_requested"]:
        raise ValueError("actual late-start timestep is outside the requested schedule range")
    if not 1 <= record["steps_run"] <= record["num_steps"]:
        raise ValueError("late-start step count is invalid")
    return record


def load_reference(config_path: Path, eval_root: Path | None) -> np.ndarray:
    for root in [eval_root, Path.cwd(), Path.cwd() / "scripts"]:
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from simdiff_eval.io import iter_real_reference_batches_from_config
    real = np.concatenate(list(iter_real_reference_batches_from_config(config_path))).astype(np.float32)
    assert real.ndim == 4 and np.isfinite(real).all(), real.shape
    return real


def unit_rows(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(len(images), -1).astype(np.float32)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("zero-norm image")
    return flat / norms


def max_cosine(samples: np.ndarray, reference: np.ndarray, chunk: int = 64) -> np.ndarray:
    ref = unit_rows(reference)
    q = unit_rows(samples)
    out = np.empty(len(q), dtype=np.float32)
    for start in range(0, len(q), chunk):
        out[start:start + chunk] = (q[start:start + chunk] @ ref.T).max(axis=1)
    return out


def boundary_ratio(images: np.ndarray) -> np.ndarray:
    """Per-image ratio: mean |neighbour jump| at 8-px patch boundaries vs. elsewhere."""
    x = images[:, 0]
    dh = np.abs(x[:, :, 1:] - x[:, :, :-1])  # jump between column j and j+1
    dv = np.abs(x[:, 1:, :] - x[:, :-1, :])
    cols = np.arange(dh.shape[2]) % PATCH == PATCH - 1
    rows = np.arange(dv.shape[1]) % PATCH == PATCH - 1
    boundary = np.concatenate([dh[:, :, cols].reshape(len(x), -1), dv[:, rows, :].reshape(len(x), -1)], axis=1)
    control = np.concatenate([dh[:, :, ~cols].reshape(len(x), -1), dv[:, ~rows, :].reshape(len(x), -1)], axis=1)
    return boundary.mean(axis=1) / control.mean(axis=1)


def radial_power(images: np.ndarray, nbins: int = 25):
    """Mean radial power per image in Fourier grid units; k bins span 0..N/2*sqrt(2)."""
    x = images[:, 0].astype(np.float64)
    n = x.shape[-1]
    f = np.fft.fft2(x)
    p = (f.real ** 2 + f.imag ** 2) / (n * n)
    kx = np.fft.fftfreq(n) * n
    kk = np.sqrt(kx[None, :] ** 2 + kx[:, None] ** 2)
    edges = np.linspace(0, kk.max(), nbins + 1)
    idx = np.clip(np.digitize(kk, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts = np.bincount(idx.ravel(), minlength=nbins)
    sums = np.stack([np.bincount(idx.ravel(), weights=pi.ravel(), minlength=nbins) for pi in p])
    return sums / np.maximum(counts, 1), centers


def score(samples: np.ndarray, reference: np.ndarray, ref_power: np.ndarray, k: np.ndarray) -> dict:
    cos = max_cosine(samples, reference)
    copies = cos > COPY_THRESHOLD
    junk = cos < JUNK_THRESHOLD
    br = boundary_ratio(samples)
    gp, _ = radial_power(samples)
    ratio = gp.mean(axis=0) / ref_power
    valid = k <= NYQUIST
    hi = (k >= NYQUIST / 2) & valid
    return {
        "n": int(len(samples)),
        "copy_fraction": float(copies.mean()),
        "junk_fraction": float(junk.mean()),
        "median_max_cos": float(np.median(cos)),
        "boundary_ratio": float(np.median(br)),
        "boundary_copies": float(np.median(br[copies])) if copies.any() else float("nan"),
        "boundary_noncopies": float(np.median(br[~copies])) if (~copies).any() else float("nan"),
        "pk_ratio_hi": float(ratio[hi].mean()),
        "pk_ratio_max_dev": float(np.abs(ratio[valid] - 1).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Run YAML defining the training subset.")
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output.")
    parser.add_argument("--eval-root", type=Path, default=None)
    args = parser.parse_args()

    reference = load_reference(args.config, args.eval_root)
    expected_training_mean_sha256 = hashlib.sha256(
        reference.mean(axis=0, keepdims=True).tobytes()
    ).hexdigest()
    ref_power, k = radial_power(reference)
    ref_power = ref_power.mean(axis=0)
    assert np.all(ref_power > 0)

    rows = []
    for path in args.samples:
        with np.load(path, allow_pickle=False) as data:
            samples = data["samples"].astype(np.float32)
            provenance = validate_provenance(
                data, training_mean_sha256=expected_training_mean_sha256
            )
        if samples.ndim != 4 or not np.isfinite(samples).all():
            raise ValueError(f"{path}: samples must be finite NCHW fields")
        if samples.shape[1:] != reference.shape[1:]:
            raise ValueError(f"{path}: sample shape {samples.shape} vs reference {reference.shape}")
        row = {"file": path.name, **provenance,
               **score(samples, reference, ref_power, k)}
        rows.append(row)
        print(" ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                       for key, value in row.items() if key != "checkpoint"))

    if args.out:
        if not rows:
            raise ValueError("no sample files were evaluated")
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        import csv
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
