# Slurm Scripts

This directory contains sanitized Slurm templates only.

Do not commit account names, user-specific home paths, scratch paths, or internal
cluster project identifiers. Copy a template into `local/` and edit it there.

`local/` is ignored by git.

## Fresh DiT-L16 300k sweep

`submit_nf_generalize_fig2_dit_l16_fresh300k.sh` trains ten DiT-L16 models
from fresh seed-123 initialization, one for every training-set size from
`2^6` through `2^15`. It runs twelve sequential 25k-update stages, for a final
budget of 300k optimizer updates per model. At most two GPU tasks run at once.

The workflow samples and analyzes the 200k checkpoint for a fixed-budget
comparison with DiT-L8 and DiT-L12, then treats the 300k checkpoint as the
primary longer-training DiT-L16 result. Exact checkpoint, sample-provenance,
PCA, and SSCD audits gate later stages so incomplete data cannot silently enter
the final curve.

Run it on Great Lakes from the repository root:

```bash
bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k.sh
```

If a stage is interrupted after writing a recovery checkpoint, resume that
stage without regenerating the frozen manifest:

```bash
START_STAGE=<stage> REUSE_EXISTING_MANIFEST=1 \
  bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k.sh
```
