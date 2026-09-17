#!/usr/bin/env python
"""CPU Test A only: same/any-location training-patch resemblance with controls.

Bounded two-axis search avoids the supplied script's ~1 GiB similarity matrix.
Nearest patches always exist, even for noise. Low majority or a large maximum
cosine is not, by itself, proof of literal patch composition or invalidity.
No model loading, sampling, Test B, training, plotting or job submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dit_a40_ablation import MATRIX_SHA, array_hash, file_hash, matrix_inputs
from patchwork_check import (corrupted_single_maps, load_reference, synthetic_patchworks,
                             to_patches, whole_image_max_cos)


def raw_patches(images):
    n, c, h, w = images.shape
    if c != 1 or h % 8 or w % 8 or not np.isfinite(images).all():
        raise ValueError("finite single-channel fields on an 8-pixel grid required")
    return images[:, 0].reshape(n, h//8, 8, w//8, 8).transpose(0, 1, 3, 2, 4).reshape(n, -1, 64)


def patch_cos_stats(samples, reference, *, query_chunk=128, bank_chunk=2048):
    if query_chunk <= 0 or bank_chunk <= 0:
        raise ValueError("chunk sizes must be positive")
    queries, ref = to_patches(samples), to_patches(reference)
    if queries.shape[1:] != ref.shape[1:]:
        raise ValueError("reference and query patch grids differ")
    same = np.empty(queries.shape[:2], dtype=np.float32)
    for p in range(queries.shape[1]):
        same[:, p] = (queries[:, p] @ ref[:, p].T).max(axis=1)
    flat, bank = queries.reshape(-1, 64), ref.reshape(-1, 64)
    best = np.full(len(flat), -np.inf, dtype=np.float32)
    selected = np.zeros(len(flat), dtype=np.int64)
    for start in range(0, len(flat), query_chunk):
        stop = min(start + query_chunk, len(flat))
        for offset in range(0, len(bank), bank_chunk):
            cos = flat[start:stop] @ bank[offset:offset+bank_chunk].T
            local = cos.argmax(axis=1)
            values = cos[np.arange(stop-start), local]
            improved = values > best[start:stop]
            best[start:stop][improved] = values[improved]
            selected[start:stop][improved] = offset + local[improved]
    # Raw-value residual, without fitting a scale/offset. Shape resemblance
    # alone can hide a patch with the wrong amplitude and mean.
    qraw = raw_patches(samples).reshape(-1, 64).astype(np.float64)
    rraw = raw_patches(reference).reshape(-1, 64)[selected].astype(np.float64)
    rms_error = np.sqrt(np.mean((qraw-rraw)**2, axis=1))
    qstd = qraw.std(axis=1)
    residual = rms_error / np.maximum(qstd, 1e-8)
    shape = queries.shape[:2]
    return same, best.reshape(shape), residual.reshape(shape)


def summarize_stats(samples, reference, mask, group):
    selected = samples[mask]
    if not len(selected):
        return {"population": group, "n": 0}
    same, anywhere, residual = patch_cos_stats(selected, reference)
    return {"population": group, "n": len(selected),
        "same_location_cos_median": float(np.median(same)),
        "any_location_cos_median": float(np.median(anywhere)),
        "any_location_cos_gt098_fraction": float((anywhere > .98).mean()),
        "raw_residual_over_patch_std_median": float(np.median(residual)),
        "max_cos_median": float(np.median(whole_image_max_cos(selected, reference)))}


def run(project, output, eval_root):
    from trace_trajectories import atomic_output, write_csv
    root, plan, baseline, _ = matrix_inputs(project)
    reference = load_reference(Path(baseline["config"]), eval_root)
    if reference.shape != (256, 1, 128, 128) or array_hash(reference) != plan["reference_sha256"]:
        raise ValueError("patch reference differs from frozen matrix training subset")
    rng = np.random.default_rng(123)
    # Phase-randomized control preserves each map's exact Fourier power, unlike
    # white noise. A huge nearest-patch bank can otherwise give a false match.
    base = reference[:32]
    phase = np.angle(np.fft.fft2(rng.normal(size=base.shape)))
    phase_randomized = np.fft.ifft2(np.abs(np.fft.fft2(base)) * np.exp(1j*phase)).real.astype(np.float32)
    controls = {"real_self_match": base,
        "synthetic_patchwork": synthetic_patchworks(reference, count=32, rng=rng),
        "corrupted_single_map_cos050": corrupted_single_maps(base, cosine=.5, rng=rng),
        "phase_randomized_real": phase_randomized,
        "gaussian_noise": (rng.normal(size=base.shape) * reference.std() + reference.mean()).astype(np.float32)}
    rows, artifacts = [], {}
    with atomic_output(output) as pending:
        for name, fields in controls.items():
            row = {"file": name, **summarize_stats(fields, reference, np.ones(len(fields), bool), "calibration")}
            rows.append(row)
            print(f"[Test A calibration] {row}", flush=True)
        for name in ("dit_l8_200k", "dit_l16_fresh300k", "dit_l16_seed456_500k"):
            directory = root / "tasks" / f"{name}__dpm50"
            report = json.loads((directory / "complete.json").read_text())
            if (report.get("status") != "complete" or report.get("plan_sha256") != MATRIX_SHA
                    or report.get("task") != {"model": name, "condition": "dpm50"}
                    or report.get("reference_sha256") != plan["reference_sha256"]):
                raise ValueError("source DPM50 task is not a completed matching matrix task")
            path = directory / "samples.npz"
            if file_hash(path) != report["artifacts_sha256"]["samples.npz"]:
                raise ValueError("source sample artifact changed")
            with np.load(path, allow_pickle=False) as data:
                fields = data["samples"].copy()
                if str(data["initial_noise_batch_sha256"].item()) != plan["noise_batch_sha256"]:
                    raise ValueError("source initial noise differs")
            if fields.shape != (128, 1, 128, 128) or not np.isfinite(fields).all():
                raise ValueError("source samples have invalid shape/values")
            cos = whole_image_max_cos(fields, reference)
            for group, mask in {"near_copy": cos > .98, "low_similarity": cos < .8,
                                 "intermediate": (cos >= .8) & (cos <= .98)}.items():
                rows.append({"file": name, **summarize_stats(fields, reference, mask, group)})
                print(f"[Test A] {rows[-1]}", flush=True)
            artifacts[str(path)] = file_hash(path)
        columns = sorted(set().union(*(r.keys() for r in rows)))
        write_csv(pending / "test_a.csv", [{key: row.get(key) for key in columns} for row in rows])
        (pending / "complete.json").write_text(json.dumps({"status": "complete", "test": "A_only",
            "source_plan_sha256": MATRIX_SHA, "source_artifacts": artifacts,
            "reference_sha256": array_hash(reference), "summary_sha256": file_hash(pending / "test_a.csv"),
            "max_similarity_workspace_bytes": 128 * 2048 * 4,
            "limitations": "Self-match is an implementation control, not held-out quality. Any-location maxima have bank-size bias. Require agreement with phase-randomized/corrupted controls and raw residuals; no causal verdict or automatic bad_fraction."}, indent=2, allow_nan=False) + "\n")
    print(f"TEST A COMPLETE: {output}; no GPU work", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("/home/jiamingp/diffusion_models_repo"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.project_dir, args.out_dir, args.eval_root)


if __name__ == "__main__":
    main()
