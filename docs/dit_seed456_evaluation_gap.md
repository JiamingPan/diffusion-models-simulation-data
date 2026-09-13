# Seed-456 evaluation: inspected gap

The local `submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh` submits only
precheck and training stages. It does not submit sampling or evaluation.

The existing `sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch`
uses the separate same-seed manifest, ten dataset tags, and its own sample root.
Do not submit it unchanged for the two seed-456 models.

The supplied final audits record 500000 updates for d2p08 and 500096 for d2p10.
They are not fresh filesystem checks. `scripts/review_dit_seed456.py` checks
the final record agreement and file presence without loading model tensors.
It rejects missing/duplicate final records and does not certify sample quality.

Before sampling, resolve the immutable runtime pin from the run provenance and
verify it with the existing runtime verifier. Select each final checkpoint from
its completion record, keep output under this experiment, and explicitly choose
raw versus EMA weights consistently with the comparison baseline. The generic
sampler supports `--ema-sigma-rel`; do not assume EMA is used by default.

Next implementation checkpoint: a seed-456-specific sampling adapter, matching
sample provenance validation, and evaluation against the original 300k baseline.
New compute and remote publication require an exact approval preview. No training
rerun is needed merely to create evaluation outputs.
