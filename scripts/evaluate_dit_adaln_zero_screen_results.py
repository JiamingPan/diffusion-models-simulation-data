#!/usr/bin/env python
"""Score one adaLN-Zero screen sample against its exact training subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

COPY_THRESHOLD = 0.98
JUNK_THRESHOLD = 0.80
PATCH = 8
NYQUIST = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(config_path: Path, eval_root: Path) -> np.ndarray:
    for root in (eval_root, eval_root / "scripts"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from simdiff_eval.io import iter_real_reference_batches_from_config
    batches = list(iter_real_reference_batches_from_config(config_path))
    real = np.concatenate(batches).astype(np.float32, copy=False)
    if real.ndim != 4 or not np.isfinite(real).all():
        raise ValueError(f"invalid reference tensor: {real.shape}")
    return real


def max_cosine_torch(samples: np.ndarray, reference: np.ndarray, device: str,
                     chunk: int = 64) -> np.ndarray:
    import torch
    ref = torch.as_tensor(reference.reshape(len(reference), -1), dtype=torch.float32)
    query = torch.as_tensor(samples.reshape(len(samples), -1), dtype=torch.float32)
    if not torch.all(torch.linalg.vector_norm(ref, dim=1) > 0):
        raise ValueError("zero-norm reference image")
    if not torch.all(torch.linalg.vector_norm(query, dim=1) > 0):
        raise ValueError("zero-norm generated image")
    ref = torch.nn.functional.normalize(ref, dim=1).to(device)
    maxima = []
    for start in range(0, len(query), chunk):
        q = torch.nn.functional.normalize(query[start:start + chunk], dim=1).to(device)
        maxima.append((q @ ref.T).amax(dim=1).cpu())
    return torch.cat(maxima).numpy()


def boundary_ratio(images: np.ndarray) -> np.ndarray:
    x = images[:, 0]
    dh = np.abs(x[:, :, 1:] - x[:, :, :-1])
    dv = np.abs(x[:, 1:, :] - x[:, :-1, :])
    cols = np.arange(dh.shape[2]) % PATCH == PATCH - 1
    rows = np.arange(dv.shape[1]) % PATCH == PATCH - 1
    boundary = np.concatenate((dh[:, :, cols].reshape(len(x), -1),
                               dv[:, rows, :].reshape(len(x), -1)), axis=1)
    control = np.concatenate((dh[:, :, ~cols].reshape(len(x), -1),
                              dv[:, ~rows, :].reshape(len(x), -1)), axis=1)
    return boundary.mean(axis=1) / control.mean(axis=1)


def mean_radial_power(images: np.ndarray, nbins: int = 25,
                      batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    n = images.shape[-1]
    kx = np.fft.fftfreq(n) * n
    kk = np.sqrt(kx[None, :] ** 2 + kx[:, None] ** 2)
    edges = np.linspace(0, kk.max(), nbins + 1)
    index = np.clip(np.digitize(kk, edges) - 1, 0, nbins - 1)
    counts = np.bincount(index.ravel(), minlength=nbins)
    total = np.zeros(nbins, dtype=np.float64)
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size, 0].astype(np.float64)
        f = np.fft.fft2(x)
        power = (f.real ** 2 + f.imag ** 2) / (n * n)
        for item in power:
            total += np.bincount(index.ravel(), weights=item.ravel(), minlength=nbins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return total / (len(images) * np.maximum(counts, 1)), centers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError("plan hash changed")
    row = json.loads(args.plan.read_text())["runs"][args.index]
    sample = Path(row["sample_path"].format(seed=123, sample_label="dpm50_n512"))
    receipt = sample.with_suffix(".complete.json")
    if not sample.is_file() or not receipt.is_file():
        raise FileNotFoundError(f"sample or completion receipt missing: {sample}")
    sample_receipt = json.loads(receipt.read_text())
    if (sample_receipt.get("status") != "complete"
            or sample_receipt.get("plan_sha256") != args.plan_sha256
            or sample_receipt.get("samples_sha256") != sha256(sample)):
        raise ValueError("sample completion receipt mismatch")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.out}")

    reference = load_reference(Path(row["config"]), args.eval_root)
    with np.load(sample, allow_pickle=False) as data:
        generated = np.asarray(data["samples"], dtype=np.float32)
    if generated.shape != (512, 1, 128, 128):
        raise ValueError(f"unexpected sample shape: {generated.shape}")

    cosine = max_cosine_torch(generated, reference, args.device)
    boundary = boundary_ratio(generated)
    real_power, k = mean_radial_power(reference)
    generated_power, _ = mean_radial_power(generated)
    ratio = generated_power / real_power
    copies = cosine > COPY_THRESHOLD
    junk = cosine < JUNK_THRESHOLD
    valid = k <= NYQUIST
    high = (k >= NYQUIST / 2) & valid
    result = {
        "run_name": row["run_name"],
        "dataset_size": row["dataset_size"],
        "n": len(generated),
        "copy_fraction": float(copies.mean()),
        "junk_fraction": float(junk.mean()),
        "intermediate_fraction": float((~copies & ~junk).mean()),
        "median_max_cos": float(np.median(cosine)),
        "boundary_ratio": float(np.median(boundary)),
        "boundary_copies": float(np.median(boundary[copies])) if copies.any() else None,
        "boundary_noncopies": float(np.median(boundary[~copies])) if (~copies).any() else None,
        "pk_ratio_hi": float(ratio[high].mean()),
        "pk_ratio_max_dev": float(np.abs(ratio[valid] - 1).max()),
        "max_cosine": cosine.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = args.out.with_suffix(args.out.suffix + ".pending")
    pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pending.replace(args.out)
    compact = {key: value for key, value in result.items() if key != "max_cosine"}
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
