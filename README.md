# Diffusion Models for CAMELS Simulation Fields

This repo evaluates diffusion models on CAMELS cosmological fields using physics-aware metrics and reproducibility/generalization diagnostics. It builds on [`nkern/cosmo_diffusion`](https://github.com/nkern/cosmo_diffusion) for base diffusion-model training and adds CAMELS-specific configs, metrics, evaluation scripts, Slurm wrappers, notebooks, and project notes.

This is not a reimplementation of `cosmo_diffusion`, and it does not claim the base training code as original work. The training entry point is still `cosmodiff_train.py` from `nkern/cosmo_diffusion`; this repo organizes experiments around it.

## Repository Layout

```text
configs/          YAML configs for CAMELS diffusion runs
scripts/          lightweight train/sample/eval wrappers and Slurm scripts
simdiff_eval/     local evaluation package for metrics and plotting
notebooks/        analysis notebooks for run sweeps
docs/             project notes and methodology explanations
results/figures/  generated figures, ignored by git except .gitkeep
results/tables/   generated metric tables, ignored by git except .gitkeep
```

Large CAMELS data files, generated samples, checkpoints, model weights, and logs are intentionally excluded from git.

## Installation

Clone this repo:

```bash
git clone git@github.com:JiamingPan/diffusion-models-simulation-data.git
cd diffusion-models-simulation-data
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Make `nkern/cosmo_diffusion` available. One simple local layout is:

```bash
git clone git@github.com:nkern/cosmo_diffusion.git cosmo_diffusion
export PYTHONPATH=$PWD/cosmo_diffusion:$PYTHONPATH
```

On Great Lakes, use the cluster Python/PyTorch environment that already works for the project, then set:

```bash
export PYTHONPATH=/home/jiamingp/Diffusion_model/cosmo_diffusion:$PYTHONPATH
```

## Example Training

Run locally or inside a Slurm job:

```bash
python scripts/train_cosmodiff.py --config configs/config_run16_u64_baseline.yaml
```

On Great Lakes, submit one of the Slurm wrappers:

```bash
sbatch scripts/slurm/train_diffusion_run16_u64_baseline.sbatch
```

These wrappers call:

```bash
python /home/jiamingp/Diffusion_model/cosmo_diffusion/scripts/cosmodiff_train.py --config <config.yaml>
```

## Example Sampling

Generate samples from a checkpoint or checkpoint directory:

```bash
python scripts/sample_cosmodiff.py \
  --checkpoint /scratch/huterer_root/huterer0/jiamingp/saved_runs/run16_u64_baseline_checkpoints \
  --output results/tables/run16_samples.npy \
  --num-samples 128 \
  --batch-size 16
```

The output `.npy` is ignored by git by default.

## Example Evaluation

Evaluate generated samples against real data loaded through a training config:

```bash
python scripts/evaluate_samples.py \
  --real-config configs/config_run16_u64_baseline.yaml \
  --generated results/tables/run16_samples.npy \
  --output-json results/tables/run16_eval.json \
  --fig-dir results/figures/run16
```

Compare multiple generated sample sets for reproducibility:

```bash
python scripts/reproducibility_eval.py \
  --generated seed1:results/tables/run16_seed1.npy \
  --generated seed2:results/tables/run16_seed2.npy \
  --output-json results/tables/run16_reproducibility.json
```

## Metrics

The evaluation scripts include:

- 2D radial power spectrum ratio, `generated P(k) / real P(k)`.
- Field one-point histogram and quantiles.
- Nearest-neighbor distance from generated images to real images in pixel space, as a simple memorization diagnostic.
- Reproducibility diagnostics across generated sample sets, including power-spectrum consistency and one-point-statistic consistency.

The notebooks add richer diagnostics such as PCA feature-space comparisons, PCA-FID/KID, image grids, and run-by-run training curves.

## Current Experiment Notes

The current configs include U64 diagnostic runs, U128/U256-width production runs, centered max-abs normalization, and CAMELS slice-thinning via `zthin`.

Important naming distinction:

- `z=0.0` in CAMELS filenames means cosmological redshift.
- `zthin` in configs means thinning the spatial depth axis of a 3D cube before converting cubes into 2D slices.

## Acknowledgements

This project builds on Nicholas Kern's `nkern/cosmo_diffusion` package for base diffusion training, checkpoint loading, data parsing, and sampling utilities. The additions here are CAMELS-focused experiment configs, wrappers, metrics, diagnostics, and analysis notebooks.

## TODO

- Add LH-data configs so training can use the full 1000-simulation CAMELS LH set instead of only the current CV subset.
- Add a documented sampling workflow that saves generated arrays for every major run.
- Add command-line PCA metrics to match the notebook diagnostics.
- Add tests for `simdiff_eval.metrics`.
- Add a small CI job for linting/import checks.
- Decide whether to track `cosmo_diffusion` as a git submodule or require it as an external dependency.
