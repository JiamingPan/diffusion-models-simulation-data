# Slurm Scripts

This directory contains sanitized Slurm templates only.

Do not commit account names, user-specific home paths, scratch paths, or internal
cluster project identifiers. Copy a template into `local/` and edit it there.

`local/` is ignored by git.

## Fresh DiT-L16 300k sweep

`submit_nf_generalize_fig2_dit_l16_fresh300k_v2.sh` trains ten independent
DiT-L16 models from seed 123, one for every training-set size from `2^6`
through `2^15`. Each Slurm task trains directly to about 300k optimizer
updates, has a 48-hour walltime, and can resume from its latest complete
recovery checkpoint. The array runs at most two GPU tasks at once.

The GPU precheck performs a real save, strict load, and resume before the full
array can start. Recovery checkpoints include model, optimizer, scheduler,
noise-scheduler, and RNG state. Only the newest two complete checkpoints are
kept during training, and the exact final checkpoint is required before
sampling. PCA and SSCD analyses run only after all ten samples finish.

Run it on Great Lakes from the repository root:

```bash
bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k_v2.sh
```

If one or more training tasks time out, submit the same frozen sweep again:

```bash
REUSE_EXISTING_MANIFEST=1 \
  bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k_v2.sh
```

Tasks that already reached the exact target exit without retraining. Other
tasks resume from their latest complete checkpoint.

The older `fresh400k` staged workflow is retained only for provenance. Do not
use it for new runs: its external checkpoint writer did not save every state
file required by the strict resume path.

### Same-checkpoint sampler audit

The focused DiT notebook uses the completed DPM50 archives as its baseline.
To test whether the fresh L16 results depend on solver length, submit the
non-destructive sampler audit:

```bash
bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit.sh
```

This creates DPM100, DPM200, and DDPM500 archives for all ten exact fresh 300k
checkpoints. The 30-task array runs at most two one-GPU tasks concurrently;
each task has 24 hours. Existing archives are skipped by default. The notebook
validates checkpoint, config, scheduler, step count, seed, and sample count
before drawing any four-sampler comparison.
