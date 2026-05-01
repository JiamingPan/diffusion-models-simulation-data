# Project Note: CAMELS Diffusion Model Evaluation

## Purpose

This repository is a research-facing layer for evaluating diffusion models on CAMELS cosmological simulation fields. The base diffusion training code comes from `nkern/cosmo_diffusion`; this repo adds CAMELS-specific experiment organization, configs, metrics, notebooks, and Slurm wrappers.

The main scientific goal is to test whether generated fields reproduce physically relevant statistics, not just whether samples look visually plausible.

## Data Representation

CAMELS grid files are stored as 3D simulation boxes. A typical configured file path looks like:

```text
Grids_HI_IllustrisTNG_CV_128_z=0.0.npy
```

Here `z=0.0` is cosmological redshift. It is not the same as the `zthin` config option.

For a raw array shaped:

```text
(N_sim, N_z, N_x, N_y)
```

the training loader can use `two_dim: true`, which converts each 3D cube into 2D slices. With `zthin: k`, the loader keeps every `k`th spatial depth slice:

```python
images = images[:, ::zthin]
images = images.reshape(-1, 1, N_x, N_y)
```

So:

```text
N_train_2D = N_sim * ceil(N_z / zthin)
```

For the current CV file with shape `(27, 128, 128, 128)`:

```text
zthin=4 -> 27 * 32 = 864 2D training images
zthin=2 -> 27 * 64 = 1728 2D training images
```

The LH CAMELS set should contain many more simulations, but the configs must point to the LH file explicitly.

## Normalization

The current stable runs use:

```yaml
log: true
minmax: false
normalization: centered_maxabs
norm_kwargs:
  center: mean
```

This first applies a log transform, subtracts the global mean, and divides by the maximum absolute value. This gives data centered near zero, which is important because the forward diffusion noise is mean-zero.

## Evaluation Philosophy

Visual inspection is useful but insufficient. The current evaluation stack checks:

- One-point field histogram and quantiles.
- 2D radial power spectrum and generated/real `P(k)` ratio.
- Nearest-neighbor distance to real fields for a simple memorization check.
- Reproducibility across independent generated sample sets.
- PCA feature-space diagnostics in notebooks.

## Relationship To `cosmo_diffusion`

This repo does not vendor or rebrand `nkern/cosmo_diffusion`. Training still calls:

```bash
cosmo_diffusion/scripts/cosmodiff_train.py --config <config.yaml>
```

The local additions are:

- Portable CAMELS config templates.
- Sanitized Slurm templates.
- Local cluster configs/scripts are intentionally excluded from git.
- Sampling/evaluation wrappers.
- Metrics in `simdiff_eval`.
- Research notebooks and notes.

## Near-Term Experimental Questions

- Does using the LH data instead of the CV subset improve `P(k)` and one-point statistics?
- Does `zthin=2` continue to help once using the larger LH dataset?
- Does U128 outperform U64 robustly when compared at similar sample count and training progress?
- Is U256 worth the GPU cost, or is it harder to optimize under current settings?
- Do generated sample sets remain statistically reproducible across seeds/checkpoints?
