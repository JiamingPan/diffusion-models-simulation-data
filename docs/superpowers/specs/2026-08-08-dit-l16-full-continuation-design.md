# DiT-L16 Full-Sweep Continuation Design

## Objective

Determine whether the anomalous DiT-L16 novelty curve, patch-like generated
structure, and power-spectrum errors are consequences of insufficient
optimization rather than a different architecture, preprocessing pipeline, or
sampler failure.

Continue every clean fresh-300k DiT-L16 run from `N_2D = 2^6` through
`N_2D = 2^15`. The experiment must produce complete, equal-budget L16
generalization curves. The `2^8` and `2^11` cases receive additional
diagnostics, but no data size receives a different training budget.

## Source Runs And Provenance

The only valid starting points are the ten completed runs in
`nf_generalize_fig2_dit_l16_fresh300k_v2`. Legacy 200k runs and the earlier
state-reset continuation are excluded.

The new experiment identity is
`nf_generalize_fig2_dit_l16_continue500k_v2`. Its manifests, configs, logs,
checkpoints, samples, and tables must use this identity and must not reuse an
output directory from any earlier sweep.

Before submission, the precheck must verify for every source run:

- the requested fresh-300k checkpoint exists and is complete;
- its stored config path and run name match the frozen manifest;
- its model, EMA, optimizer, learning-rate scheduler, scaler, and RNG recovery
  state can be loaded;
- its recorded optimizer step is at least 300k and is consistent with the
  checkpoint epoch arithmetic;
- the continuation output directory cannot collide with any existing legacy
  sweep.

The continuation must preserve the complete training state. It must not reset
AdamW moments, the cosine-restart scheduler, EMA state, mixed-precision scaler,
or random-number generators.

## Architecture And Preprocessing Invariants

The existing sweep generator specifies the same tokenization for DiT-L8,
DiT-L12/base, and DiT-L16:

- image size: 128 by 128;
- patch size: 8 by 8;
- token grid: 16 by 16;
- input and output channels: one;
- attention heads: 12;
- attention-head dimension: 64;
- normalization groups: 32.

The only model-field difference is transformer depth: 8, 12, or 16 layers.

The continuation precheck must load the exact frozen L8, L12, and fresh-L16
configs used on Great Lakes and compare all architecture, data,
normalization, transform, scheduler, and sampling fields. It must fail on any
unexpected difference. Allowed differences are model depth, run identity,
checkpoint paths, dataset size, and the explicitly labeled optimizer budget.

## Training Schedule

Continue all ten L16 runs from 300k to 500k optimizer updates. Use one Slurm
array with at most two GPU tasks active at once.

Evaluate at the following total-update targets:

- 300k: completed baseline;
- 340k;
- 380k;
- 420k;
- 460k;
- 500k.

The 40k spacing keeps every evaluation point at the same phase of the
4k-update cosine warm-restart schedule. Labels and tables must record the
actual optimizer step reached, not only the nominal target.

The nominal DPM sample labels are `dpm50_fresh300k_v2`,
`dpm50_cont340k_v2`, `dpm50_cont380k_v2`, `dpm50_cont420k_v2`,
`dpm50_cont460k_v2`, and `dpm50_cont500k_v2`. Stored provenance must also
record the actual optimizer step, so an epoch-rounded checkpoint cannot be
mistaken for an exact nominal target.

Each task receives 48 hours and writes complete recovery checkpoints about
every 5k optimizer updates. A restarted task resumes from the latest complete
recovery checkpoint. Downstream sampling uses `afterok` dependencies and must
not run for a failed or incomplete target.

## Sampling Design

At every evaluation point, generate 512 samples for each of the ten data
sizes with DPM-Solver at 50 steps. Reuse the same seed schedule and initial
noise across checkpoints so changes can be attributed to training rather than
different random draws.

The sampling audit must record:

- requested and resolved checkpoint;
- config path and sample label;
- scheduler class and configuration;
- requested and executed inference-step count;
- first and final scheduler timesteps;
- confirmation that the terminal update produces the scheduler's zero-noise
  output state;
- seed schedule and generated tensor shape.

For `2^8` and `2^11`, run matched DDPM-500 controls at 300k and 500k. If the
patch structure or power-spectrum error is present under both DPM-50 and
DDPM-500, treat it as a model/training result rather than a fast-sampler
artifact.

## Analysis

### Generalization Curves

Compute PCA and SSCD q95 novelty for all ten L16 data sizes at every evaluation
point. Plot:

1. L16-only curves from 300k through 500k;
2. the final L16 curve against L8 and L12/base;
3. the final L16 curve against existing UNet references.

Every unequal-budget comparison must label optimizer budgets explicitly. The
analysis must report the observed curves without selecting a checkpoint merely
because it follows an expected capacity-scaling pattern.

### Physical Statistics

For every data size and evaluation point, compute:

- one-point PDF error using common real/generated bins;
- mean generated-to-real power-spectrum ratio;
- log-ratio power-spectrum error;
- per-sample power at bins 20, 40, and 60 using a fixed set of 100 generated
  samples;
- sample variance, standard deviation, and bootstrap confidence interval at
  those three bins.

Each black real reference must use the exact training subset configured for
that run. Tables must record its config path and observed slice count.

### Patch-Grid Diagnostics

Quantify the visible block structure rather than relying on image inspection.
For real maps and generated maps from L8, L12, and L16, compute a patch-boundary
ratio: the mean absolute horizontal and vertical gradient across every eighth
pixel divided by the corresponding mean away from patch boundaries.

A value near the real-data baseline indicates no excess patch discontinuity.
An elevated value isolated to L16 supports an undertrained patch-token
representation even though all depths use the same patch size. Plot this score
against data size and optimizer updates. Include selected two-dimensional
Fourier views only if they reveal a reproducible grid-frequency excess.

### Diagnostic Image Panels

Create a final 500k generated-map grid across all ten data sizes. For `2^8`
and `2^11`, also show matched samples across optimizer checkpoints and sampler
types. Preserve sample identity across columns by using the same seed and
sample index.

## Notebook Integration

Update the existing
`notebooks/nf_generalize_fig2_dit_results.ipynb`; do not create another primary
results notebook. Add compact provenance and missing-artifact audits before
plotting. Figures must remain separate enough to avoid overlapping labels and
must not silently substitute a legacy sample or another data size.

## Failure Handling

- Abort before GPU training if config parity or checkpoint recovery fails.
- Never fall back from an expected checkpoint to the latest directory.
- Never analyze a sample whose stored checkpoint or config does not match the
  manifest.
- Preserve completed array tasks when rerunning failures.
- Report missing checkpoints, samples, and PCA/SSCD tables explicitly.
- Keep expensive real-reference loading streamed or batched.

## Verification

- Unit-test update-to-epoch arithmetic for all ten dataset sizes.
- Test that every continuation config differs from its fresh-300k source only
  in permitted training and output fields.
- Test complete-state resume and a short optimizer-step round trip.
- Test exact-checkpoint sampling and stored provenance.
- Test the sampler terminal-state audit with DPM-50 and DDPM-500.
- Test the patch-boundary metric on smooth images and synthetic 8-pixel block
  artifacts.
- Compile every notebook code cell and validate notebook JSON.
- Execute the full workflow on Great Lakes and visually inspect the resulting
  curves and image grids.

## Acceptance Criteria

The experiment is complete when all ten clean L16 runs reach the final target,
all final samples and PCA/SSCD tables pass provenance checks, and the existing
results notebook produces complete L16 generalization, physical-statistics,
power-variance, and patch-boundary curves. The evidence must distinguish among
longer-training improvement, persistent model error, patch-grid artifacts, and
fast-sampler artifacts without assuming in advance that L16 must follow the
L8/L12 scaling relation.
