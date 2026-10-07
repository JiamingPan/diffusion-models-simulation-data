#!/usr/bin/env python
"""Manifests for the memorization-regime sampler test (no training).

Resamples the existing Omega_m-sigma_8 models at N = 64 and 256 with more accurate
samplers than the baseline DPM-Solver 50 steps, to test whether the smoothed copies
(and the probe bias they cause) come from the sampler. Same checkpoints, same
held-out cosmologies, 64 draws each. New sample paths only; existing samples untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local" / "memorization_sampler_test"
SWEEPS = ("nf_conditional_omsig_continuous_200k", "nf_conditional_omsig_class36_200k")
SIZES = (64, 256)
# tag -> (diffusers scheduler, steps). DDPM 500 = the training scheduler at full length.
SAMPLERS = {"ddpm500": ("DDPMScheduler", 500), "dpm200": ("DPMSolverMultistepScheduler", 200)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for tag, (scheduler, steps) in SAMPLERS.items():
        rows = []
        for sweep in SWEEPS:
            for row in json.loads((ROOT / "local" / sweep / "manifest.json").read_text()):
                if row["dataset_size"] not in SIZES:
                    continue
                new = dict(row, source_sweep=sweep, sampler_tag=tag, scheduler=scheduler, num_steps=steps,
                           sample_path=f"results/memorization_regime_diagnostics/sampler_test/samples/{row['run_name']}_seed{{seed}}_{tag}_heldout_k{{k}}.npz")
                rows.append(new)
                tasks.append({"task": len(tasks), "sampler_tag": tag, "run_name": row["run_name"], "scheduler": scheduler, "num_steps": steps})
        (OUT / f"manifest_{tag}.json").write_text(json.dumps(rows, indent=1))
    (OUT / "tasks.json").write_text(json.dumps(tasks, indent=1))
    for t in tasks:
        print(t)


if __name__ == "__main__":
    main()
