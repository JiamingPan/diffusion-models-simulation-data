# DiT-L16 Resume Target Fix Design

## Problem

The DiT-L16 continuation workflow mixed two meanings of
`train.num_epochs`. The manifest generator wrote an absolute final epoch, but
the Great Lakes `cosmodiff_train.py` resume path treated that value as a number
of additional epochs. A run intended to stop at epoch 14,062 therefore resumed
at epoch 12,792 and stopped at epoch 26,854. The inner trainer completed
normally, but the Slurm wrapper rejected the run because the exact requested
checkpoint did not exist.

Previous attempts also wrote overshot checkpoints into the original training
directories. Those directories can no longer provide an uncontaminated
200k-to-300k checkpoint trajectory.

## Required Behavior

- Preserve existing checkpoints; do not delete or rewrite them.
- Create one isolated continuation checkpoint directory for each DiT-L16 run
  from `d2p06` through `d2p10`.
- Seed each continuation directory with the exact original 200k checkpoint.
- Run four sequential stages of exactly 25,000 additional optimizer updates.
- End at nominal totals of 225k, 250k, 275k, and 300k updates.
- Sample and analyze only each stage's exact checkpoint.
- Refuse to train when the selected starting checkpoint or update arithmetic
  does not match the manifest.

## Checkpoint Layout

The original training directories remain read-only inputs. Continuation
checkpoints are written under a separate root, with one directory per run. The
stage-1 directory is seeded with a symlink to the exact 200k checkpoint. Later
stages resume from the exact preceding stage checkpoint in the same isolated
directory.

No continuation job may discover checkpoints from the original, contaminated
directory after selecting the 200k seed.

## Epoch And Update Arithmetic

For each run:

1. Compute optimizer updates per epoch from the configured dataset size,
   micro-batch size, and gradient accumulation.
2. Identify the exact original 200k checkpoint epoch.
3. Convert 25,000 additional optimizer updates into the stage's additional
   epoch count.
4. Before training, read the current continuation checkpoint and assert that
   it equals the expected previous-stage epoch.
5. Pass the additional epoch count required by the Great Lakes trainer.
6. Predict the resulting final epoch and assert that it equals the manifest's
   exact stage target before launching the expensive process.
7. After training, require the exact target checkpoint to exist.

The workflow must fail before GPU training if any arithmetic or checkpoint
identity assertion fails.

## Failure Recovery

Safety checkpoints may be written within a stage. A restarted stage may resume
from a safety checkpoint only after recomputing the remaining epochs to the
same exact stage target. It must never add the full stage duration again.

If a safety checkpoint is already beyond the target, the stage fails with a
clear contamination error. It does not relabel or sample the overshot model.

## Tests

Regression tests will reproduce the observed resume case:

- resume epoch: 12,792;
- requested final epoch: 14,062;
- incorrect old result: 26,854;
- required remaining epochs: 1,271.

Tests will also cover clean-directory seeding, stage-to-stage targets,
mid-stage recovery, overshoot rejection, exact-checkpoint sampling, and shell
syntax. The focused continuation and checkpoint-resume suites must pass before
the workflow is submitted again.

## Migration

The frozen manifest from the failed workflow will not be reused. A new manifest
and configs will be generated for the isolated checkpoint root. Existing
overshot checkpoints remain available for debugging but are excluded from all
new continuation discovery, sampling, and analysis.
