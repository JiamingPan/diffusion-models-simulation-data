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

The project builds on the training and data utilities in
[`nkern/cosmo_diffusion`](https://github.com/nkern/cosmo_diffusion), but it is
not only a collection of notebooks around that package. The DiT integration,
conditional experiments, sampling and resume code, augmentation support,
cluster workflows, and evaluation framework used in this study are implemented
here.

## What This Repository Contains

### Model and training extensions

- **Diffusion transformers:** this repository adds the
  [`DiTTransformer2DModel`](https://huggingface.co/docs/diffusers/api/models/dit_transformer2d)
  path used in the project. It generates DiT configurations, passes the labels
  required by adaptive layer normalization through training, loads DiT
  checkpoints directly, runs depth sweeps, and resumes staged training at exact
  checkpoints.
- **U-Net experiments:** the U-Net workflow covers unconditional models,
  class-conditional models, and continuous conditioning on the six CAMELS
  cosmology and feedback parameters.
- **Sampling:** the checkpoint-aware sampler loads both U-Net and DiT models,
  reconstructs the saved noise scheduler, supports post-hoc EMA, records sample
  provenance, and can replace the training scheduler with DPM-Solver.
- **Physics-preserving augmentation:** the augmentation path adds random
  90-degree rotations and the eight symmetries of a square, and fixes flip
  dimensions for scalar field maps.
- **Long-run safeguards:** preflight checks, saved manifests, exact checkpoint
  targets, and resume guards are used before expensive training and sampling
  jobs are submitted.

Key implementation files:

- [DiT sweep and model configuration](scripts/prepare_nf_generalize_fig2_dit_configs.py)
- [DiT class-label and checkpoint integration](scripts/patch_cosmodiff_dit_class_labels.py)
- [Exact DiT checkpoint resume](scripts/run_cosmodiff_train_with_dit_resume.py)
- [U-Net and DiT sampling](scripts/sample_cosmodiff.py)
- [Continuous conditional U-Net configuration](scripts/prepare_nf_conditional_u128_config.py)
- [Square-symmetry augmentations](scripts/patch_cosmodiff_augmentations.py)

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
scripts/            model integration, config, training, sampling, and analysis
scripts/slurm/      Great Lakes training, sampling, and analysis workflows
notebooks/          reproducible analysis and figure notebooks
tests/              resume, continuation, reference-data, and notebook checks
docs/               methodology, experiment, and workflow notes
results/            local tables and figures; large outputs are mostly ignored
```

Representative entry points:

- [PCA nearest-neighbor analysis](scripts/compute_nf_generalize_pca_full_nn.py)
- [SSCD nearest-neighbor analysis](scripts/compute_nf_generalize_sscd_full_nn.py)
- [Conditional parameter-recovery evaluation](scripts/evaluate_nf_conditional_bias_probe.py)
- [Staged DiT continuation workflow](scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue.sh)
- [Fresh DiT-L16 400k sweep](scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh400k.sh)

More detailed command references are available in
[`scripts/README.md`](scripts/README.md),
[`configs/README.md`](configs/README.md), and
[`notebooks/README.md`](notebooks/README.md).

## Notebook Index

All tracked analysis notebooks are linked below. The executed DiT snapshot is
kept separately so collaborators can inspect its saved outputs without rerunning
the full analysis.

### Memorization, generalization, and physical statistics

- [SSCD generalizability figure](notebooks/generalizability_figure.ipynb)
- [PCA generalizability figure](notebooks/generalizability_figure_pca.ipynb)
- [PCA representation diagnostics](notebooks/generalizability_figure_pca_diagnostics.ipynb)
- [U-Net data-size sweep and physical-statistics checks](notebooks/nf_generalize_fig2_partial_quickcheck.ipynb)
- [Small-data memorization quickcheck](notebooks/nf_generalize_n64_memorize_quickcheck.ipynb)
- [Reference U-Net data sweep](notebooks/nf_generalize_nick_data_quickcheck.ipynb)
- [Reference sweep with SSCD and physical statistics](notebooks/nf_generalize_nick_data_sscd_quickcheck.ipynb)
- [DDPM and DPM-Solver sampling comparison](notebooks/nf_generalize_nick_data_dpm50_compare.ipynb)
- [Transition-scaling diagnostic](notebooks/nf_generalize_scaling_diagnostic.ipynb)
- [Training-checkpoint learning process](notebooks/nf_generalize_epoch_snapshots_poster.ipynb)

### Diffusion transformers

- [DiT depth and data-size results](notebooks/nf_generalize_fig2_dit_results.ipynb)
- [Executed DiT results snapshot](notebooks/nf_generalize_fig2_dit_results_executed.ipynb)

### Conditional generation and calibration

- [Four-field class-conditional quickcheck](notebooks/nf_class_conditional_fourfield_quickcheck.ipynb)
- [Continuous conditional inference](notebooks/nf_conditional_inference_results.ipynb)
- [Continuous cosmology bias probe](notebooks/nf_conditional_bias_probe_check.ipynb)
- [VGG cosmology recovery results](notebooks/nf_conditional_bias_vgg_results.ipynb)

### Training, EMA, augmentation, and ablations

- [U64 run and hyperparameter inspection](notebooks/inspect_run16_run12_run15.ipynb)
- [U64, U128, and U256 run inspection](notebooks/inspect_run9_run10_run11.ipynb)
- [U64 full-data EMA check](notebooks/nf_generalize_fig2_u64_d2p15_ema_check.ipynb)
- [Poster EMA and guidance ablations](notebooks/nf_poster_ablation_appendix.ipynb)
- [Augmentation sweep quickcheck](notebooks/nf_sweep_aug_quickcheck.ipynb)
- [EMA length sweep quickcheck](notebooks/nf_sweep_ema_sigma_quickcheck.ipynb)
- [Initial sweep inspection](notebooks/nf_sweep_inspection.ipynb)
- [Small-EMA quickcheck](notebooks/nf_sweep_small_ema_quickcheck.ipynb)
- [Direct and post-hoc EMA comparison](notebooks/nf_sweep_v2_ema_direct_vs_posthoc.ipynb)
- [Second sweep quickcheck](notebooks/nf_sweep_v2_quickcheck.ipynb)
- [Second sweep smoke-training check](notebooks/nf_sweep_v2_smoke_train_check.ipynb)

### Reproducibility and data validation

- [Normalization-fix and SSCD evaluation](notebooks/normalization_fixes_sscd_eval.ipynb)
- [Cross-run reproducibility analysis](notebooks/reproducibility_figure_analysis.ipynb)

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
the original diffusion training and CAMELS data utilities. The DiT integration,
conditional-model experiments, augmentation support, model-aware sampling and
resume logic, sweep automation, and evaluation framework described above are
implemented in this repository.
