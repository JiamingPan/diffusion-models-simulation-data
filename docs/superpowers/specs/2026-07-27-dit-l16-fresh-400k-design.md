# Fresh DiT-L16 400k Sweep Design

## Goal

Train ten DiT-L16 models from fresh initialization, one for each training-set
size from \(2^6\) through \(2^{15}\), to 400,000 requested optimizer updates.
The experiment must determine the measured L16 novelty transition without
assuming that it lies to the right of DiT-L12.

## Isolation

The sweep uses the identity `nf_generalize_fig2_dit_l16_fresh400k`, run names
ending in `_fresh400k_seed123`, and new checkpoint, sample, table, log, and
local-manifest roots. It must not read from the historical continuation sweep
or the superseded `fresh300k` sweep.

All ten runs use the same DiT-L16 architecture and training recipe. Python,
NumPy, PyTorch CPU, and PyTorch CUDA RNGs are seeded with 123 before model and
data-loader construction. Resume stages restore the model, optimizer,
learning-rate scheduler, noise scheduler, and saved RNG state.

## Training And Evaluation

Training is divided into sixteen sequential 25k-update stages. Each array task
uses one GPU, and the array permits at most two concurrent GPU tasks. Recovery
checkpoints are written approximately every 5,000 updates. A 24-hour walltime
is used for each 25k stage, and rerunning the same stage resumes from its latest
valid recovery checkpoint.

Full sampling, PCA analysis, SSCD analysis, and provenance auditing occur only
at:

- 200k: equal-budget comparison with DiT-L8 and DiT-L12;
- 300k: intermediate longer-training diagnostic;
- 400k: primary final DiT-L16 result.

Each evaluated checkpoint produces 512 samples using
`DPMSolverMultistepScheduler`, 50 solver steps, sampling seed 123, and class
label 0.

## Notebook Contract

The DiT results notebook must:

- require exactly ten dataset sizes for every displayed fresh milestone;
- show 200k as the fixed-budget comparison;
- label 300k as intermediate;
- use 400k as the primary final L16 curve;
- provide full-range and transition-zoom views;
- show the 200k, 300k, and 400k L16 trajectories;
- withhold the final curve if either the PCA or SSCD 400k table is incomplete;
- never substitute legacy continuation or `fresh300k` values;
- state that a rightward shift is a hypothesis, not an enforced outcome.

## Acceptance

The manifest contains 160 rows: ten data sizes times sixteen stages. Every
stage-16 row targets 400,000 requested updates. Scientific milestones are
exactly 200k, 300k, and 400k. Shell scripts parse, Python files and notebook
cells compile, the notebook updater is idempotent, and the full test suite
passes.
