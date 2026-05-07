# Diffusion Models for CAMELS Simulation Fields

This repo evaluates diffusion models on CAMELS cosmological fields using physics-aware metrics and reproducibility/generalization diagnostics. It builds on [`nkern/cosmo_diffusion`](https://github.com/nkern/cosmo_diffusion) for base diffusion-model training and adds CAMELS-specific metrics, evaluation scripts, templates, notebooks, and project notes.

This is not a reimplementation of `cosmo_diffusion`, and it does not claim the base training code as original work. The training entry point is still `cosmodiff_train.py` from `nkern/cosmo_diffusion`; this repo organizes experiments around it.

## Repository Layout

Reusable layer:

```text
simdiff_eval/       local evaluation package for metrics and plotting
scripts/*.py        lightweight train/sample/eval wrappers
configs/templates/  portable CAMELS LH config templates
scripts/slurm/      sanitized Slurm template
notebooks/          analysis notebooks for run sweeps
docs/               project notes and methodology explanations
```

Output layer:

```text
results/figures/    generated figures, ignored by git except .gitkeep
results/tables/     generated metric tables, ignored by git except .gitkeep
```

Large CAMELS data files, generated samples, checkpoints, model weights, logs, personal cluster paths, and account-specific Slurm files are intentionally excluded from git. For new runs, start from `configs/templates/` and `scripts/slurm/train_template.sbatch`, then edit paths locally.

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

On a cluster, use a Python/PyTorch environment compatible with `cosmo_diffusion`, then set:

```bash
export PYTHONPATH=/path/to/cosmo_diffusion:$PYTHONPATH
```

## Example Training

Run locally or inside a Slurm job using a template or edited config:

```bash
python scripts/train_cosmodiff.py --config configs/templates/u64_lh_template.yaml
```

For Slurm, copy and edit the sanitized template:

```bash
cp scripts/slurm/train_template.sbatch local/my_train.sbatch
# edit local/my_train.sbatch
sbatch local/my_train.sbatch
```

The template calls:

```bash
python /path/to/cosmo_diffusion/scripts/cosmodiff_train.py --config <config.yaml>
```

## Example Sampling

Generate samples from a checkpoint or checkpoint directory:

```bash
python scripts/sample_cosmodiff.py \
  --checkpoint /path/to/checkpoints \
  --output results/tables/generated_samples.npy \
  --num-samples 128 \
  --batch-size 16
```

The output `.npy` is ignored by git by default.

## Example Evaluation

Evaluate generated samples against real data loaded through a training config:

```bash
python scripts/evaluate_samples.py \
  --real-config configs/templates/u64_lh_template.yaml \
  --generated results/tables/generated_samples.npy \
  --output-json results/tables/eval.json \
  --fig-dir results/figures/example
```

Compare multiple generated sample sets for reproducibility:

```bash
python scripts/reproducibility_eval.py \
  --generated seed1:results/tables/run16_seed1.npy \
  --generated seed2:results/tables/run16_seed2.npy \
  --output-json results/tables/run16_reproducibility.json
```

Make a Figure-1-style pairwise reproducibility plot across model widths:

```bash
python scripts/plot_reproducibility_figure.py \
  --manifest local/reproducibility_manifest.json \
  --sample-root results/tables/samples \
  --output-csv results/tables/reproducibility_scores.csv \
  --output-figure results/figures/reproducibility_scores.png
```

For real experiments, copy `configs/templates/reproducibility_manifest_template.json`
to an ignored local path and edit the run names, dataset sizes, and generated
sample paths.

## Metrics

The evaluation scripts include:

- 2D radial power spectrum ratio, `generated P(k) / real P(k)`.
- Field one-point histogram and quantiles.
- Nearest-neighbor distance from generated images to real images in pixel space, as a simple memorization diagnostic.
- Reproducibility diagnostics across generated sample sets, including power-spectrum consistency and one-point-statistic consistency.

The notebooks add richer diagnostics such as PCA feature-space comparisons, PCA-FID/KID, image grids, and run-by-run training curves.

The reproducibility-figure script uses a transparent project score:

```text
error = P(k) log10 MAE between generated sets
      + absolute difference in field mean
      + absolute difference in field standard deviation

score = 1 / (1 + error)
```

This gives a compact score in `(0, 1]`, where larger means the two generated
sample sets are more statistically similar under these diagnostics. It is a
Figure-1-style diagnostic, not a claim to exactly match another paper's score
unless that score is implemented separately.

Compute a paper-style SSCD generalizability curve for one model width:

```bash
python scripts/plot_generalizability_sscd.py \
  --arch u64 \
  --config-dir local/fig1_lh/configs \
  --sample-root results/tables/samples \
  --sscd-path /path/to/sscd_disc_mixup.torchscript.pt \
  --output-csv results/tables/generalizability_sscd_u64.csv \
  --output-figure results/figures/generalizability_sscd_u64.png
```

This score follows the near-copy logic used in SSCD-based generalizability
plots: a generated image is counted as memorized if its maximum SSCD cosine
similarity to any real training image exceeds the threshold. Keep this separate
from P(k): SSCD is a copy/generalization diagnostic, while P(k) is a
physics-fidelity diagnostic. See `docs/sscd_generalizability_note.md` for the
equations and the reason P(k)-nearest-neighbor scores are not equivalent to
SSCD copy detection.

## Current Experiment Notes

The templates include U64, U128, and U256-width starting points with centered max-abs normalization and CAMELS slice-thinning via `zthin`.

Important naming distinction:

- `z=0.0` in CAMELS filenames means cosmological redshift.
- `zthin` in configs means thinning the spatial depth axis of a 3D cube before converting cubes into 2D slices.

## Acknowledgements

This project builds on Nicholas Kern's `nkern/cosmo_diffusion` package for base diffusion training, checkpoint loading, data parsing, and sampling utilities. The additions here are CAMELS-focused experiment configs, wrappers, metrics, diagnostics, and analysis notebooks.

## TODO

- Add sanitized example configs for the reproducibility/data-size experiment after the final run plan is fixed.
- Add a documented sampling workflow that saves generated arrays for every major run.
- Add command-line PCA metrics to match the notebook diagnostics.
- Add tests for `simdiff_eval.metrics`.
- Add a small CI job for linting/import checks.
- Decide whether to track `cosmo_diffusion` as a git submodule or require it as an external dependency.
