# Seed-Restart Constant-Label Contract

## Goal

Make the DiT-L16 seed-restart wrapper accept an already-correct constant-label
dataset without weakening the protection against replacing genuine conditional
labels.

## Required behavior

When `data.constant_label` is configured, the wrapper must distinguish exactly
three cases:

1. `dataset.labels is None`: inject a one-dimensional `torch.long` tensor whose
   length matches the image count and whose values equal `constant_label`.
2. Labels already exist, their length matches the image count, and every value
   equals `constant_label`: preserve the existing tensor unchanged.
3. Labels already exist but their length or values conflict: print provenance
   and raise before training.

Every path must print one flushed provenance line containing the path name,
label dtype, length, and unique values. The immutable pin manifest must state
whether support existed in the unmodified base revision, whether the published
pin supports it, and whether that support is native or supplied by the declared
constant-label patch.

## Scope and safety

- Preserve the real-label guard.
- Do not modify source checkpoints.
- Do not touch C4 code or results.
- Do not push or submit Slurm jobs.
- Test the three cases before implementation.
- Audit compatibility from `parse_config_data` through the first optimizer step.
- Finish with the full test suite, compile checks, shell syntax checks, and
  `git diff --check`.

## Verified root cause

The base Cosmodiff revision `58c77eb45de6e4d135697ba83ffee93ae54d918c`
does not contain native `data.constant_label` handling. The immutable pin
builder applies `patch_cosmodiff_constant_label.py`, so the published pin does
contain the handling before the in-process adapter runs. The adapter then sees
the patch-created labels and mistakes them for genuine conditional labels.
