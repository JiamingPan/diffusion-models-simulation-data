# Diffusion Models for CAMELS Simulation Fields

This repository studies diffusion models for CAMELS cosmological simulation fields. The main question is whether a model learns a useful distribution of physical fields, or whether it mainly reproduces the finite training set.

The project builds on [`nkern/cosmo_diffusion`](https://github.com/nkern/cosmo_diffusion) for the base diffusion training code. This repository adds CAMELS-specific experiment organization, evaluation scripts, diagnostics, notebooks, and project notes. It is an active research repository; the public README is intentionally high level while the analysis is still being cleaned up for a paper.

## Project Goals

- Train diffusion models on CAMELS 2D field slices derived from 3D simulation volumes.
- Measure when generated samples are near-copies of training fields versus genuinely new samples.
- Check whether generated fields remain physically meaningful using one-point statistics and power spectra.
- Test conditional generation: ask for a cosmology, generate an HI field, then recover the cosmology from the generated field with an independent encoder.

## Repository Layout

```text
simdiff_eval/       reusable evaluation and plotting utilities
scripts/            lightweight training, sampling, preparation, and analysis wrappers
scripts/slurm/      sanitized Slurm templates used for cluster runs
configs/templates/  portable example configs
notebooks/          analysis and figure-making notebooks
docs/               project notes, experiment summaries, and methodology notes
results/            generated tables and figures; mostly ignored or symlinked locally
```

Large CAMELS data files, generated samples, checkpoints, personal paths, logs, and account-specific Slurm files are intentionally excluded from git.

## Scope

This is a working research codebase rather than a polished benchmark package. The repo contains the reusable analysis code, templates, and selected notebooks needed to understand the workflow, but it does not publish raw data, trained checkpoints, private cluster paths, or every intermediate run artifact. Quantitative claims should be treated as current working results until the manuscript version is released.

## Minimal Setup

Clone this repository and install the Python dependencies:

```bash
git clone git@github.com:JiamingPan/diffusion-models-simulation-data.git
cd diffusion-models-simulation-data
python -m pip install -r requirements.txt
```

The base training code comes from `nkern/cosmo_diffusion`, which should be available on `PYTHONPATH` or installed in the active environment. Most real experiments were run on a GPU cluster, but this README intentionally avoids account-specific commands.

## Typical Workflow

At a high level, the workflow is:

1. Prepare a CAMELS field config and choose training-set size.
2. Train a diffusion model using `cosmo_diffusion`.
3. Sample generated fields from checkpoints, often with DPM-Solver for faster inference.
4. Evaluate generated fields with physical statistics and memorization/generalization diagnostics.
5. Inspect and polish results in notebooks.

Representative notebooks currently used for figures and checks are:

- `notebooks/nf_generalize_fig2_partial_quickcheck.ipynb`: memorization/generalization curves and physical-statistics checks.
- `notebooks/nf_conditional_bias_probe_check.ipynb`: conditional-cosmology calibration checks.
- `notebooks/nf_generalize_scaling_diagnostic.ipynb`: exploratory scaling diagnostics for the transition.

## Current Results

These are current working results, not final paper claims.

- **Memorization-to-generalization transition:** generated fields are close to individual training slices at small data sizes and become less training-set-like as data increases.
- **Feature-space diagnostics:** the transition is checked with nearest-neighbor comparisons in multiple image/field representations, plus reproducibility checks across independently trained models.
- **Physical fidelity checks:** generated fields are compared to real fields using one-point pixel-value distributions and power spectra, so the evaluation is not based only on visual similarity.
- **Conditional cosmology calibration:** for HI-only continuous conditioning, generated fields are encoded back to cosmological parameters with an independent frozen encoder. The high-data regime tracks the requested matter-density parameter more reliably than the small-data memorization regime.
- **Exploratory scaling:** early diagnostics suggest the transition shifts with model capacity, but the current evidence is too sparse to treat the fitted scaling law as a final result.

## Key Evaluation Ideas

- **One-point PDF:** checks whether generated field values follow the same marginal distribution as real fields.
- **Power spectrum `P(k)`:** checks whether generated fields reproduce spatial structure across scales.
- **Nearest-neighbor similarity:** checks whether generated samples are too close to training samples.
- **Reproducibility:** compares generated sets from independently trained models.
- **Conditional calibration:** compares requested cosmology parameters with parameters recovered from generated fields using an encoder trained only on real fields.

## Status

This repository is work in progress for an ongoing research project and poster/workshop-paper preparation. The code is useful for reproducing the current analysis workflow, but paths, notebooks, and experiment names may still change as the project is cleaned up.

## Acknowledgements

This project builds on Nicholas Kern's `nkern/cosmo_diffusion` package for base diffusion training, checkpoint loading, data parsing, and sampling utilities. The additions here are CAMELS-focused experiment configs, wrappers, diagnostics, and analysis notebooks.
