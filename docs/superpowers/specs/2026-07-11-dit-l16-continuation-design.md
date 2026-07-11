# DiT-L16 Small-Data Continuation Design

## Objective

Test whether the physically invalid small-data DiT-L16 samples are caused by
insufficient optimization. Continue the existing `d2p06` through `d2p10` runs
without changing architecture, data, optimizer, scheduler, EMA, or seed.

## Training Design

- Continue five existing DiT-L16 checkpoint directories.
- Add 100,000 optimizer updates in four sequential 25,000-update stages.
- Produce evaluation checkpoints at nominal totals of 225k, 250k, 275k, and
  300k updates.
- Save internal safety checkpoints approximately every 5,000 updates.
- Give each stage eight hours. Stages are submitted with `afterok`
  dependencies, so later stages cannot start after a failure.
- Re-running a failed stage resumes from its latest safety checkpoint.

The config generator reads the latest checkpoint epoch and converts optimizer
updates to epochs using the DiT micro-batch size and gradient accumulation.
It writes absolute final epoch counts because `cosmodiff_train.py` resumes from
the latest checkpoint and interprets `train.num_epochs` as an absolute limit.

## Sampling And Evaluation

After each stage, sample the exact stage-final checkpoint with 50-step
DPM-Solver, 512 samples, seed 123, and a stage-specific sample label. Evaluate:

- generated image grids;
- one-point PDFs;
- mean power-spectrum ratios;
- PCA q95 novelty;
- SSCD q95 novelty.

The acceptance criterion is a consistent movement of the power-spectrum ratio
toward one across checkpoints. Lower denoising loss alone is not improvement.

## Reference-Data Correction

The UNet-128 controlled fidelity comparison currently uses the correct run
configuration but caps the real reference at 16 of 64 training simulations.
For this controlled comparison, load the complete configured training set and
assert that its number of 2D slices equals `dataset_size` (1,024 for `d2p10`).
Record the config path and slice count in the comparison table.

## Failure Handling

- Missing base checkpoints are fatal before training starts.
- A stage config records the checkpoint epoch observed during preparation.
- Each training task validates model depth, v-prediction, null label, and
  gradient accumulation before launching.
- Safety checkpoints limit work lost to roughly 5,000 updates.
- Sampling requires the exact expected final checkpoint; it never silently
  samples the latest directory.

## Verification

- Unit-test stage arithmetic and config invariants.
- Static-test Slurm walltime, safety interval, and exact-checkpoint sampling.
- Validate notebook JSON and compile every code cell.
- Run the existing DiT runtime precheck before submitting continuation jobs.
