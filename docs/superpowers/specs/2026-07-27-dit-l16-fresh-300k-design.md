# Fresh DiT-L16 300k Sweep Design

## Purpose

Run a new DiT-L16 memorization-to-generalization sweep from random
initialization. The experiment must replace neither the original 200k sweep nor
the legacy 225k--300k continuation outputs. Its purpose is to determine whether
the unusual low-data DiT-L16 behavior persists when all training sizes are
trained reproducibly with complete training-state recovery.

**Every one of the ten fresh DiT-L16 runs trains from initialization through
300,000 requested optimizer updates. The sweep does not stop at 200k.** The
200k checkpoint is retained as an intermediate matched-budget comparison with
the existing L8 and L12 runs; 300k is the final checkpoint of the new L16
experiment.

The experiment tests a hypothesis; it does not tune the implementation until
the DiT-L16 curve appears to the right of DiT-L12. A high nearest-neighbor
novelty score is not accepted as generalization unless the generated maps also
pass image and physical-statistics checks.

## Decision

Use a fresh, isolated DiT-L16 sweep covering all ten training-set sizes. Train
each run to approximately 300,000 optimizer updates in twelve sequential
25,000-update stages. Stage boundaries are operational only: every resume must
restore the model, AdamW moments, learning-rate scheduler, noise scheduler, and
random-number-generator state.

Rejected alternatives:

1. Continue the old 200k checkpoints. Those checkpoints already give suspect
   low-data outputs, and the previous special continuation path reset optimizer
   and scheduler state.
2. Retrain only `2^6`--`2^10`. That cannot produce a complete DiT-L16
   generalization curve or establish where the transition saturates.
3. Tune until the transition follows the expected capacity trend. That would
   bias the experiment toward the desired conclusion.

## Experiment Identity and Isolation

- Sweep name: `nf_generalize_fig2_dit_l16_fresh300k`
- Training seed: `123`
- Sampling seed: `123`
- Run names:
  `nf_fig2_dit_l16_fresh300k_s123_<dataset_tag>_noaug`
- Dataset tags:
  `d2p06`, `d2p07`, `d2p08`, `d2p09`, `d2p10`, `d2p11`, `d2p12`,
  `d2p13`, `d2p14`, and `d2p15`
- Checkpoint root:
  `/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_generalize_fig2_dit_l16_fresh300k`
- Local configuration root:
  `local/nf_generalize_fig2_dit_l16_fresh300k`
- Result root:
  `results/nf_generalize_fig2_dit_l16_fresh300k`

The preparer must fail if a run directory already contains a checkpoint unless
the operator explicitly selects resume mode. No old checkpoint may be linked,
copied, or selected by a "latest checkpoint" search.

## Fixed Model and Training Configuration

Every run uses the existing DiT-L16 architecture:

- `DiTTransformer2DModel`
- image size `128`
- patch size `8`
- one input and one output channel
- `16` transformer layers
- `12` attention heads
- head dimension `64`, giving hidden width `768`
- one unconditional null class with label `0`

The training configuration remains fixed across dataset sizes:

- no augmentation
- micro-batch size `2`
- gradient accumulation `4`
- effective batch size `8`
- AdamW, learning rate `1e-4`, weight decay `1e-2`
- `CosineAnnealingWarmRestarts`, `T_0=4000`, `eta_min=1e-7`
- fp16 mixed precision
- gradient clipping at `1.0`
- EMA update every step after a 1,000-update burn-in
- min-SNR gamma `5.0`
- 500-step cosine VP noise schedule
- zero-terminal-SNR rescaling
- v-prediction

The data seed and process seed are both `123`. The fresh-training launcher must
seed Python, NumPy, PyTorch CPU, and every CUDA device before the dataset,
model, optimizer, or scheduler is created.

## Data Provenance

The ten training subsets use the existing CAMELS source-allocation logic and
must be deterministic and nested where the current allocator supports nesting.
Each manifest row records:

- source paths and source counts
- `n_samples`
- `zthin`
- normalization and transforms
- data seed
- training seed
- model configuration
- cosmo_diffusion branch and commit
- repository commit

PCA, SSCD, one-point, and power-spectrum references must be loaded from the
exact config for that run. The analysis may not use one common empirical
reference for all training sizes. References can look similar because the
subsets sample the same CAMELS population, but their provenance remains
separate.

## Update Targets and Checkpoints

The scientific comparison checkpoints are approximately:

- 200k updates
- 225k updates
- 250k updates
- 275k updates
- 300k updates

Training runs in twelve 25k target stages from zero to 300k. Checkpoints are
saved at most 5k optimizer updates apart so a timeout loses no more than about
5k updates. Because the external trainer saves on epoch boundaries, each
manifest row records both requested and actual optimizer updates. Figures use
the actual value in captions and may use `200k`--`300k` as short labels only
when the discrepancy is explicitly recorded.

No checkpoint is removed automatically. The workflow produces a dry-run
retention report after a stage succeeds. Deleting recovery checkpoints requires
separate explicit approval.

## Stateful Resume Contract

A resumed stage must fail before allocating a GPU training loop unless all of
the following hold:

1. The current checkpoint belongs to the fresh sweep and the expected run.
2. Its epoch is not beyond the stage target.
3. The diffusers class is `DiTTransformer2DModel`.
4. Model weights contain no meta tensors.
5. `optimizer.pkl` exists and contains nonempty AdamW state.
6. `lr_scheduler.pkl` exists and is rebound to the restored optimizer.
7. `noise_scheduler.pkl` exists.
8. `random_states_0.pkl` exists and its Python, NumPy, PyTorch, and CUDA states
   can be restored.
9. The exact stage-start checkpoint and target checkpoint agree with the frozen
   manifest.

The first stage uses a fresh-training wrapper and rejects any existing
checkpoint. Later stages use the exact-target resume wrapper. Re-running a
completed stage is a no-op after validating its exact target checkpoint.

## Cluster Workflow

- Training and sampling each request one GPU.
- At most two training or sampling array tasks run concurrently.
- Each training task requests 24 hours and 80 GB RAM.
- Each sampling task requests 4 hours and 80 GB RAM.
- PCA and SSCD analyses run on the standard partition.
- The workflow submits stage dependencies sequentially.
- A failed stage prevents all downstream sampling, analysis, and training
  stages from starting.

Before the full sweep, one GPU precheck must:

1. build all ten configs and validate the manifest,
2. verify that every data source is readable,
3. run a labeled DiT forward/backward/optimizer step,
4. save and reload a tiny checkpoint,
5. prove that model, optimizer, scheduler, noise scheduler, and RNG state are
   restored,
6. resume for one update and confirm that counters advance exactly once.

## Sampling and Evaluation

At 200k, 225k, 250k, 275k, and 300k, sample all ten dataset sizes with:

- DPM-Solver multistep scheduler
- 50 sampling steps
- 512 generated fields
- sampling seed `123`
- the exact manifest checkpoint, never a directory-wide latest-checkpoint
  lookup

Every sample file records the requested checkpoint, resolved checkpoint,
config path, scheduler, number of steps, seed, training seed, requested
updates, and actual updates.

For every scientific checkpoint, compute:

- PCA q95 nearest-neighbor novelty
- SSCD q95 nearest-neighbor novelty
- representative generated maps
- generated-versus-training nearest-neighbor panels
- one-point PDF against that run's real subset
- mean generated-to-real power-spectrum ratio

## Notebook and Figure Contract

Update `notebooks/nf_generalize_fig2_dit_results.ipynb` to provide:

1. a file and provenance audit,
2. a primary fixed-budget depth comparison using DiT-L8 200k, DiT-L12 200k,
   and the fresh DiT-L16 200k checkpoint across all ten data sizes,
3. complete DiT-L16 novelty curves at each analyzed checkpoint,
4. a separate L16 optimization comparison using 200k, 225k, 250k, 275k, and
   300k, without presenting it as a fixed-capacity scaling plot,
5. a final-outcome comparison showing DiT-L8 200k, DiT-L12 200k, and fresh
   DiT-L16 300k, explicitly labeled as an unequal-update comparison,
6. a full-range `2^6`--`2^15` panel,
7. a transition-region zoom panel that still shows every available point,
8. a companion transition-location plot showing the interpolated q95
   `N_50` against DiT depth and trainable parameter count for L8, L12, and
   fresh L16 at 200k,
9. map grids and physical-statistics panels for the five L16 checkpoints,
10. checkpoint trajectories for each dataset size.

The fixed-budget depth figure is the primary scaling diagnostic. It uses one
line per DiT depth, common axes, distinct colors and markers, a visible 0.5
reference, and separate PCA and SSCD panels. The figure caption states that all
depths use the same 200k-update budget and that high novelty alone does not
establish physical validity. The full-range and zoom views use the same data;
the zoom changes only the displayed x range.

Before treating the L8/L12/L16 comparison as controlled, the notebook audits
that architecture depth is the intended difference and that data allocation,
optimizer, scheduler, noise schedule, batch size, gradient accumulation, EMA,
sampling configuration, and update budget match. If the historical L8 or L12
training seed is not recoverable, the figure is labeled an exploratory
fixed-budget comparison rather than a precise scaling-law measurement.

The notebook must refuse to label a curve complete unless all ten dataset sizes
are present for both PCA and SSCD. It must not splice old and fresh DiT-L16
points into one line. Legacy continuation tables remain visible only in an
explicitly invalid-results audit.

## Interpretation Rules

The result supports a capacity-dependent transition only if:

- the full ten-point L16 curve is available,
- the fixed-budget L8/L12/fresh-L16 comparison passes its configuration audit,
- the transition location is stable in both PCA and SSCD,
- maps around the transition remain physically plausible,
- one-point and power-spectrum checks do not identify off-distribution noise,
- the conclusion does not depend on one checkpoint chosen after inspection.

If 300k low-data runs remain physically poor, the result is reported as an
optimization or model failure, not as successful generalization. If the L16
transition does not move right of L12, that result is retained rather than
retuned away. With only three DiT depths, the transition-location plot is a
capacity diagnostic, not evidence for a universal scaling law.

## Verification

Local tests cover:

- fresh run naming and output isolation,
- all ten dataset sizes,
- deterministic seeds,
- update and epoch arithmetic,
- complete-state checkpoint restoration including RNG state,
- exact checkpoint selection for sampling,
- dependency cancellation,
- complete-table requirements in the notebook.

Great Lakes verification requires:

- the precheck to pass,
- one short smoke stage to create and resume a real DiT checkpoint,
- the full array to be submitted only after the smoke stage succeeds.
