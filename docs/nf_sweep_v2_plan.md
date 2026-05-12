# NF Sweep V2 Plan

`nf_sweep_v2` is an additive sweep.  It does not replace the already-running
`nf_sweep` jobs.  It tracks Nicholas Kern's merged `cosmo_diffusion/main`
defaults after the `normalization_fixes` branch was merged.

## What Changed Versus `nf_sweep`

- Uses `cosmo_diffusion/main`, not the deleted `normalization_fixes` branch.
- Uses the new data API:

```yaml
data:
  reshape: '2d'
  transform: [log]
```

instead of:

```yaml
data:
  two_dim: true
  log: true
```

- Uses Nicholas's merged scheduler/training default:

```yaml
noise_scheduler:
  kwargs:
    beta_schedule: squaredcos_cap_v2
    rescale_betas_zero_snr: true
    prediction_type: v_prediction

train:
  ema_sigma_rels: [0.02, 0.10]
  ema_update_every: 1
  ema_burn_in: 1000
  min_snr_gamma: 5.0
```

- Adds the new `generate:` config block and sample jobs for fast inference
  schedulers.

## Variants

Each architecture gets seven runs:

- `nick_default`: Nicholas's merged default.
- `sigmoid_zero_snr`: swaps cosine beta schedule for sigmoid while keeping
  zero-SNR and Min-SNR.
- `cosine_no_zero_snr`: ablates `rescale_betas_zero_snr`.
- `no_minsnr`: ablates `min_snr_gamma`.
- `no_ema_burnin`: changes `ema_burn_in` from `1000` to `0`.
- `sigma_edm`: adds `sigma_log_normal: [-1.2, 1.2]`.
- `sigma_edm2`: adds `sigma_log_normal: [-0.4, 1.0]`.

The default architectures are U64 and U128, both with `n_samples=500`,
`zthin=4`, and `num_epochs=100`.  With the current 128-slice cube files,
that is `500 * (128 / 4) = 16000` 2D training slices per run.

## Great Lakes Usage

Set up the current main checkout and generate configs:

```bash
cd /home/jiamingp/diffusion_models_repo
git pull
unset COSMODIFF_DIR
bash scripts/setup_cosmodiff_main.sh
python scripts/prepare_nf_sweep_v2_configs.py --project-dir "$PWD"
python scripts/prepare_nf_sweep_v2_configs.py --project-dir "$PWD" --check-only
```

Smoke test one U64 run:

```bash
sbatch --array=0-0 --time=00:20:00 scripts/slurm/train_nf_sweep_v2_array.sbatch
```

Full v2 training array:

```bash
sbatch scripts/slurm/train_nf_sweep_v2_array.sbatch
```

The v2 scripts use account `huterer2` by default.

## Sampling

The sample array writes `.npz` files to:

```text
results/nf_sweep_v2/samples/
```

It samples each run for:

- EMA targets: raw, 0.02, 0.04, 0.06, 0.08, 0.10
- inference schedulers: training scheduler/full steps, `DPMSolverMultistepScheduler` at 25 steps,
  and `HeunDiscreteScheduler` at 50 steps

Submit sampling after checkpoints exist:

```bash
sbatch scripts/slurm/sample_nf_sweep_v2_array.sbatch
```
