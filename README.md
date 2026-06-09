# Diffusion Models for CAMELS Simulation Fields

This repository studies diffusion models for CAMELS cosmological simulation fields. The main question is whether a diffusion model learns the underlying distribution of physical fields, or whether it memorizes the finite training set.

The project builds on [`nkern/cosmo_diffusion`](https://github.com/nkern/cosmo_diffusion) for the base diffusion training code. This repository adds CAMELS-specific experiment organization, evaluation scripts, diagnostics, notebooks, and project notes. It is an active research repo, so results and interfaces are still evolving.

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

The main notebooks currently used for figures and checks are:

- `notebooks/nf_generalize_fig2_partial_quickcheck.ipynb`: memorization/generalization curves, one-point checks, power spectra, and poster figures.
- `notebooks/nf_conditional_bias_probe_check.ipynb`: continuous-cosmology calibration checks.
- `notebooks/nf_poster_ablation_appendix.ipynb`: guidance/CFG ablation plots for appendix-style checks.

## Current Results

These are current working results, not final paper claims.

- **Memorization-to-generalization transition:** generated fields are training-set-like at small dataset sizes and become less training-set-like as training data increases. This is measured with nearest-neighbor diagnostics in PCA and SSCD feature spaces.
- **Architecture dependence:** wider U-Net models generally require more data and/or training to reach the same generalization behavior.
- **Physical fidelity checks:** generated fields are compared to real fields using one-point pixel-value distributions and radial power spectra, so the evaluation is not based only on visual similarity.
- **Faster sampling:** DPM-Solver multistep sampling with 50 steps gives much faster generation than the original 500-step DDPM baseline while preserving the key diagnostics used here.
- **Conditional cosmology calibration:** for HI-only continuous conditioning, generated fields are encoded back to cosmology parameters. The best current probe uses frozen VGG16 features with average+max pooling and an MLP regression head. The large-data model tracks the requested `Omega_m` much better than the small-data model; `sigma_8` is weaker, and feedback parameters remain harder to recover from HI alone.

## Key Evaluation Ideas

- **One-point PDF:** checks whether generated field values follow the same marginal distribution as real fields.
- **Power spectrum `P(k)`:** checks whether generated fields reproduce spatial structure across scales.
- **Nearest-neighbor similarity:** checks whether generated samples are too close to training samples.
- **Reproducibility:** compares generated sets from independently trained models.
- **Conditional calibration:** compares requested cosmology parameters with parameters recovered from generated fields.

## Status

This repository is work in progress for an ongoing research project and poster/workshop-paper preparation. The code is useful for reproducing the current analysis workflow, but paths, notebooks, and experiment names may still change as the project is cleaned up.

## Acknowledgements

This project builds on Nicholas Kern's `nkern/cosmo_diffusion` package for base diffusion training, checkpoint loading, data parsing, and sampling utilities. The additions here are CAMELS-focused experiment configs, wrappers, diagnostics, and analysis notebooks.
