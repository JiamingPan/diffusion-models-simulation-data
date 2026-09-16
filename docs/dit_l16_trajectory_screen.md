# Paired DiT trajectory screen

## Scope and completion

No training, no EMA, and no sampler matrix. Run L8 200k, L16 fresh300k,
and L16 seed456 500k at N2D=256 sequentially in one GPU allocation.
L12 is explicitly omitted: its selected checkpoint has no raw model weights.
Do not change checkpoint/config contents or the immutable runtime pin.

The screen is complete only when all three per-model terminal JSON reports
exist and `comparison/complete.json` confirms identical saved initial noise,
training-reference order/content, seed, and timesteps across the three models.
The wrapper prints its final COMPLETE line only after this comparison succeeds.
Existing results are never overwritten. Failed temporary output is retained
with no terminal PASS. A killed writer can leave a `.lock` file; investigate
and confirm the writer is gone before asking for authorization to remove it.

## Protocol

- Ordinary deterministic DPMSolverMultistepScheduler, 50 steps, raw weights.
- 32 draws, seed 123, null class 0, model-forward chunks of 8.
- CPU float32 initial Gaussian noise, unchanged on transfer to GPU. Each
  sample's exact initial noise and sha256 is saved and checked across models.
  These are new paired draws, not matching indices in older GPU-RNG exports.
- Record the model's x0 estimate before each scheduler update, using the
  v-prediction identity. Advance the solver once per timestep for the whole batch.
- Record majority, persistence, distinct maps, boundary ratio, nearest-training
  max cosine, top1-top2 global gap, x0 RMS, and median best local patch cosine.
- Save all stepwise patch assignments, and x/x0 fields at selected steps.
- Persistence is undefined at the first step, not zero. CSV uses an empty cell.
- Outcome labels retain the existing strict thresholds: copy >0.98,
  junk <0.8, other in between. Junk is a screening label, not physical validity.

## Outputs and inspection

Under `results/dit_l16_trajectory_screen/<model>/`:

- `<model>_trajectory.csv`: step medians split by that model's final outcome.
- `<model>_trajectory.npz`: per-sample arrays, assignments, retained fields,
  final fields, outcomes, exact initial noise and hashes.
- `<model>_trajectory.json`: terminal provenance, outcome counts, unrounded
  final max-cosine list, scheduler settings, input/artifact checksums.
- `<model>_trajectory.png` and `<model>_gallery.png`: Pillow-only figures;
  no import of Matplotlib or its failing system C++ extension.
- `persistence_controls.csv`: independent, correlated and frozen Gaussian
  noise, frozen corrupted single maps, and frozen synthetic patchworks.

Under `comparison/`:

- `paired_outcomes.csv`: one row per noise draw, all three final outcomes/cosines.
- `all_trajectories.csv`: the three requested per-model CSVs concatenated.
- `paired_by_l16_outcome.csv`: both L16 outcome masks applied to exactly the
  same sample indices in ALL models. Use this table to compare L8 on L16-junk
  draws, not L8's aggregate copies versus a different L16 subset.
- `complete.json`: terminal cross-model validation.

Read counts and paired sample outcomes, then persistence together with majority,
local patch cosine, global gap, and boundary ratio around SNR 0.1-0.5.
The curves use chronological solver steps (early left, late right); the gray
band denotes the recorded steps in that SNR range. Individual curves are faint,
and outcome medians are thick. Galleries use a fixed [-1,1] display range.

## Interpretive limits

Low majority plus high persistence is evidence of stable local assignments,
not unique evidence of copied tiles. Frozen Gaussian noise and a frozen noisy
single map also give persistence 1. Correlated successive fields can give
high persistence without copying. The local-fit control helps check whether
stable assignments are also close matches; neither metric proves causality.

Check the actual x0 galleries and saved sampler states against these controls.
A successful denoiser's high-noise x0 estimate need not already be a coherent
sample. Different training durations also prevent this from being a pure
depth-only intervention. Outcome grouping is descriptive, not a causal cohort.
Even the predicted pattern does not prove stochastic sampling fixes the issue;
that would require the separately approved sampler intervention afterward.

## Runtime and external actions

Use `scripts/slurm/evaluate_dit_trajectories.sbatch`. Set the same frozen
runtime code root `iaifi_poster_code_555f350` and cosmodiff pin used by the
successful v2 probe. The adapter CODE_ROOT is separate; do not repoint the
pin to it. The stub directory is first on PYTHONPATH and user site is disabled.

`PREFLIGHT_ONLY=1 bash .../evaluate_dit_trajectories.sbatch` checks runtime,
Pillow, exact code roots, inputs and CLI without GPU inference, result writes,
or job submission. Model weights are loaded only inside the allocation.

Local tests/edits are autonomous. Push requires an exact preview followed by
APPROVE PUSH. Slurm submission requires its own exact preview followed by
APPROVE RUN. Never infer permission from prior approvals. No C4 changes.
