# Diffusion Models for CAMELS Simulation Fields

## Overview

This repository studies diffusion models as generators for cosmological
simulation fields. The main question is whether a model trained on a finite
simulation set learns the underlying field distribution or mainly reproduces
individual training examples.

The experiments use two-dimensional slices of neutral-hydrogen fields from the
CAMELS IllustrisTNG simulations. The code supports unconditional U-Net and
diffusion transformer (DiT) experiments, plus U-Net models conditioned on
continuous cosmological parameters. It also provides tools for comparing model
families, training-set sizes, and checkpoints.

The training code builds on
[`nkern/cosmo_diffusion`](https://github.com/nkern/cosmo_diffusion). This
repository adds experiment preparation, portable configuration templates,
sampling wrappers, evaluation code, Great Lakes workflows, and analysis
notebooks.

## What This Repository Contains

### Model training and sampling

- U-Net and DiT configuration templates for unconditional models, plus
  continuous conditioning for U-Net experiments.
- Saved manifests that keep data-size and architecture sweeps fixed and
  reviewable.
- Checkpoint continuation and resume checks for long-running experiments.
- Sampling with standard diffusion schedulers, including DPM-Solver for faster
  generation from trained checkpoints.
- Slurm job chains for training, sampling, and downstream analysis on Great
  Lakes.

### Memorization and sample-novelty diagnostics

- Nearest-neighbor comparisons between generated and training fields.
- Similarity measurements in pixel, PCA, and SSCD representation spaces.
- Cross-run comparisons between independently trained models. These checks ask
  whether separate runs reproduce the same individual fields or agree only at
  the level of the overall distribution.
- Adaptive thresholds based on nearest-neighbor distances among real fields,
  rather than a single hand-selected similarity cutoff.

### Physical and conditional evaluation

- One-point field-value distributions.
- Power spectra for comparing spatial structure across scales.
- Conditional parameter-recovery tests using encoders trained only on real
  simulation fields.
- VGG16-based calibration checks that compare a requested cosmology with the
  cosmology recovered from generated fields.

These tests are kept separate on purpose. A generated field can be novel while
having incorrect physical statistics, or it can match summary statistics while
responding incorrectly to conditioning parameters.

## Evaluation Framework

The analysis addresses three different questions:

1. **Does the model copy its training data?** Generated fields are compared
   with their nearest training examples in several representation spaces.
2. **Does the generated distribution reproduce relevant field statistics?**
   One-point distributions and power spectra compare generated and real
   ensembles.
3. **Does conditioning have the intended physical effect?** A frozen encoder
   trained on real fields estimates cosmological parameters from generated
   fields and tests whether those estimates track the requested values.

No single metric is treated as proof that a model is suitable for inference.
Pixel, PCA, and SSCD similarity answer different questions, while physical
statistics and parameter recovery test behavior that nearest-neighbor scores
do not measure.

## Workflows and Repository Map

A typical experiment follows this sequence:

1. Generate a model configuration and a manifest for the selected data size
   and architecture.
2. Train the model and save checkpoints, including EMA state where configured.
3. Generate samples from specific checkpoints with a recorded scheduler,
   number of sampling steps, and random seed.
4. Run nearest-neighbor, physical-statistics, and conditional-calibration
   analyses.
5. Inspect aggregate results and figures in a notebook.

```text
configs/templates/  portable U-Net and DiT configuration templates
simdiff_eval/       reusable statistics, nearest-neighbor, and plotting code
scripts/            config preparation, sampling, and evaluation entry points
scripts/slurm/      Great Lakes training, sampling, and analysis workflows
notebooks/          reproducible analysis and figure notebooks
docs/               methodology, experiment, and workflow notes
results/            local tables and figures; large outputs are mostly ignored
```

Representative entry points:

- [DiT sweep preparation](scripts/prepare_nf_generalize_fig2_dit_configs.py)
- [Conditional U-Net preparation](scripts/prepare_nf_conditional_u128_config.py)
- [Checkpoint-aware sampling](scripts/sample_cosmodiff.py)
- [PCA nearest-neighbor analysis](scripts/compute_nf_generalize_pca_full_nn.py)
- [SSCD nearest-neighbor analysis](scripts/compute_nf_generalize_sscd_full_nn.py)
- [Conditional parameter-recovery evaluation](scripts/evaluate_nf_conditional_bias_probe.py)
- [U-Net generalization and physical-statistics notebook](notebooks/nf_generalize_fig2_partial_quickcheck.ipynb)
- [DiT depth and data-size notebook](notebooks/nf_generalize_fig2_dit_results.ipynb)
- [Conditional calibration notebook](notebooks/nf_conditional_bias_probe_check.ipynb)
- [Transition-scaling diagnostics](notebooks/nf_generalize_scaling_diagnostic.ipynb)

More detailed command references are available in
[`scripts/README.md`](scripts/README.md),
[`configs/README.md`](configs/README.md), and
[`notebooks/README.md`](notebooks/README.md).

## Setup

Clone the repository and install its Python dependencies:

```bash
git clone git@github.com:JiamingPan/diffusion-models-simulation-data.git
cd diffusion-models-simulation-data
python -m pip install -r requirements.txt
```

The base `cosmo_diffusion` package must also be installed or available on
`PYTHONPATH`. CAMELS data, generated samples, and checkpoints are not included
in this repository. Most full experiments require a CUDA GPU and were run
through Slurm, while the reusable analysis modules can also be used outside the
cluster environment.

## Research Status

This is an active research repository, not a finished benchmark package. The
current code supports controlled U-Net and DiT comparisons and the complete
evaluation workflow described above. Ongoing studies examine how the observed
memorization-to-generalization behavior depends on training-set size, model
architecture and depth, training duration, EMA, augmentation, and sampling
choices.

Results from DiT depth sweeps and model-capacity scaling are still exploratory.
They are included as analysis tools and working notebooks rather than stated as
final scaling laws. Large datasets, private cluster paths, raw checkpoints, and
intermediate run artifacts are intentionally excluded from git.

## Acknowledgements

This project builds on Nicholas Kern's
[`nkern/cosmo_diffusion`](https://github.com/nkern/cosmo_diffusion) package for
the core diffusion training, data parsing, checkpoint, and sampling utilities.
The additions here focus on CAMELS experiment design, cluster workflows,
memorization diagnostics, physical-statistics checks, and conditional
calibration.
