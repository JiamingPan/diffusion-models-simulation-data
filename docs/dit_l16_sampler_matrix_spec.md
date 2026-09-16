# DiT-L16 paired sampler matrix

## Requirements

- No training, EMA, C4 edits, checkpoint edits, main-branch changes, or deletion.
- N2D=256: L8 200k and both L16 checkpoints; include L12 only if its exact
  selected 200k checkpoint has raw config/weights and the matching run config.
- Raw weights, null class 0, model-forward chunks of 8, 128 draws. Preserve
  the original 32 CPU-float32 initial fields byte-for-byte as the first 32.
- Compare DPM++ 50, DPM++ 200, DDPM 500, and SDE-DPM++ 50/order 2. Record
  actual scheduler configuration, steps, and separate stochastic RNG provenance.
- Read the already-completed outcome-conditioned trajectory CSV and galleries;
  do not repeat that GPU experiment.

## Completion

Continue through local tests, native CPU scheduler checks, compile/shell checks,
and diff verification. Then, after the external-action gates, run the frozen
Great Lakes preflight, one sampler array, and the CPU-only aggregate read.
Complete only when every planned arm has checksummed samples and scores and
the aggregate verifies all 128 pairs, including the original 32. A successful
job is not proof of sample quality or a causal mechanism.

Runtime, exact commands, outputs, and interpretation rules are in
[the runtime and validation guide](dit_l16_sampler_matrix_runtime.md).
Push/merge/upload requires an exact preview and **APPROVE PUSH**; submitting
or cancelling jobs requires a separate exact preview and **APPROVE RUN**.
Approvals are one-time. Never infer them from earlier experiments.
