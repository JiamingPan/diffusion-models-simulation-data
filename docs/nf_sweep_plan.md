# Normalization-Fix Diffusion Sweep

This sweep starts from the completed `nf_u64_n500_e100` and
`nf_u128_n500_e100` runs, then changes one training choice at a time.

The EDM2 paper (`2312.02696v2`) motivates two parts of this sweep:

- Log-normal noise-level sampling: EDM used `P_mean=-1.2, P_std=1.2`; EDM2
  reports `P_mean=-0.4, P_std=1.0` for latent ImageNet experiments.
- Post-hoc EMA: train two EMA profiles once, then synthesize multiple EMA
  lengths at sampling/evaluation time instead of retraining.

The sweep intentionally does not include U256 yet.  The first pass is U64 and
U128 with `n_samples=500`, `zthin=4`, `num_epochs=100`, and the same tanh data
normalization used by the existing normalization-fix runs.

## Definitions

This section defines the diffusion terms used below.

Forward noising process:

```text
alpha_t = 1 - beta_t
alpha_bar_t = product(alpha_1, ..., alpha_t)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
```

Here `x_0` is a real training image/slice, `noise` is Gaussian noise, and
`x_t` is the noised image at diffusion timestep `t`.

`beta_schedule` controls the sequence of `beta_t` values.  Larger `beta_t`
means more noise is added at that timestep.  Changing `beta_schedule` changes
the whole forward noising curve, so it is a training-time change.

`SNR` means signal-to-noise ratio:

```text
SNR_t = alpha_bar_t / (1 - alpha_bar_t)
```

Large `SNR_t` means the image is still mostly signal.  Small `SNR_t` means the
image is mostly noise.  At the final timestep, ideal pure noise has
`alpha_bar_t = 0`, so `SNR_t = 0`.

`rescale_betas_zero_snr: true` modifies the beta schedule so the final training
timestep has zero terminal SNR.  This is meant to reduce train/inference
mismatch: sampling starts from pure Gaussian noise, so the noisiest training
timestep should also be pure noise.  `timestep_spacing: trailing` makes
inference start from the last/noisiest timestep.

`prediction_type` controls what the model is trained to predict:

- `epsilon`: predict the Gaussian noise added to the image.
- `v_prediction`: predict the velocity target used by the scheduler.  In
  simplified form, this mixes the clean image and noise as
  `v_t = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x_0`.

`sigma` is another way to parameterize noise level:

```text
sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t)
log_sigma_t = log(sigma_t)
```

`sigma_log_normal: [P_mean, P_std]` does not change the beta schedule.  It
changes how training timesteps are sampled.  Instead of sampling timesteps
uniformly, the code samples:

```text
log_sigma ~ Normal(P_mean, P_std)
```

and maps that noise level back to the nearest discrete DDPM timestep.  This
focuses training on particular noise levels.  `[-1.2, 1.2]` is the EDM default;
`[-0.4, 1.0]` is the EDM2 latent-image setting.  For CAMELS HI fields these are
experimental choices, not guaranteed optima.

`min_snr_gamma` changes the loss weighting across timesteps.  Without it, every
sample's MSE contributes equally.  With `min_snr_gamma: 5.0`, very high-SNR
timesteps are capped so they cannot dominate training.  In this branch, for
`v_prediction` the per-sample MSE is weighted roughly by:

```text
min(SNR_t, gamma) / (SNR_t + 1)
```

For `epsilon`, it is weighted roughly by:

```text
min(SNR_t, gamma) / SNR_t
```

`EMA` means exponential moving average of model weights, not noise levels.  The
post-hoc EMA implementation stores two weight averages during training:

```yaml
ema_sigma_rels: [0.03, 0.25]
```

These two values are basis EMA profiles.  After training, the sampler can
synthesize intermediate EMA lengths such as `0.05`, `0.08`, `0.13`, and `0.20`
without retraining.  Smaller EMA length is closer to the final raw weights;
larger EMA length averages over more of the training trajectory.

## Variants

Each architecture gets these runs:

- `base_ema`: old sigmoid + `v_prediction` baseline, retrained with post-hoc
  EMA enabled.  This is the control run for the sweep.
- `zero_snr`: old sigmoid + `v_prediction`, plus
  `rescale_betas_zero_snr: true` and `timestep_spacing: trailing`.  This checks
  whether forcing the final timestep to pure noise improves the model.
- `beta_linear`: changes only `beta_schedule` to `linear`.  This checks whether
  the old standard DDPM beta curve beats the current sigmoid curve.
- `beta_cosine`: changes only `beta_schedule` to `squaredcos_cap_v2`.  This
  checks the cosine-style schedule from improved DDPM implementations.
- `pred_eps`: changes only `prediction_type` to `epsilon`.  This tests whether
  direct noise prediction is better or worse than `v_prediction`.
- `sigma_edm`: keeps sigmoid + `v_prediction`, adds
  `sigma_log_normal: [-1.2, 1.2]`.  This checks the EDM default timestep
  sampling distribution.
- `sigma_edm2`: keeps sigmoid + `v_prediction`, adds
  `sigma_log_normal: [-0.4, 1.0]`.  This checks the EDM2 latent-image timestep
  sampling distribution.
- `minsnr5`: keeps sigmoid + `v_prediction`, adds `min_snr_gamma: 5.0`.  This
  checks whether capped SNR loss weighting improves training.

All new runs use:

```yaml
train:
  ema_sigma_rels: [0.03, 0.25]
  ema_update_every: 1
  ema_burn_in: 0
```

The two trained EMA profiles bracket the target EMA values sampled later:
raw weights, 0.03, 0.05, 0.08, 0.13, and 0.20.

## Great Lakes Usage

Generate configs only:

```bash
cd /home/jiamingp/diffusion_models_repo
git pull
bash scripts/setup_cosmodiff_normalization_fixes.sh
unset COSMODIFF_DIR
python scripts/prepare_nf_sweep_configs.py --project-dir "$PWD"
python scripts/prepare_nf_sweep_configs.py --project-dir "$PWD" --check-only
```

The setup script clones/updates the `normalization_fixes` branch into
`/home/jiamingp/Diffusion_model/cosmo_diffusion_normalization_fixes_git`,
installs `ema-pytorch` into the active environment if it is missing, and checks
that Python imports `cosmodiff` from the git checkout.

Submit training only:

```bash
sbatch scripts/slurm/train_nf_sweep_array.sbatch
```

Submit sampling only after training finishes:

```bash
sbatch scripts/slurm/sample_nf_sweep_array.sbatch
```

Submit both with dependency:

```bash
bash scripts/slurm/submit_nf_sweep.sh
```

The sample array writes 64 samples per run/EMA target by default.  Override
with environment variables when submitting, for example:

```bash
NUM_SAMPLES=128 BATCH_SIZE=8 sbatch scripts/slurm/sample_nf_sweep_array.sbatch
```

For faster smoke-test sampling with DDIM:

```bash
NUM_SAMPLES=8 DDIM_THINNING=10 sbatch scripts/slurm/sample_nf_sweep_array.sbatch
```
