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

## Variants

Each architecture gets these runs:

- `base_ema`: old sigmoid + `v_prediction` baseline, retrained with post-hoc
  EMA enabled.
- `beta_linear`: changes only `beta_schedule` to `linear`.
- `beta_cosine`: changes only `beta_schedule` to `squaredcos_cap_v2`.
- `pred_eps`: changes only `prediction_type` to `epsilon`.
- `sigma_edm`: keeps sigmoid + `v_prediction`, adds
  `sigma_log_normal: [-1.2, 1.2]`.
- `sigma_edm2`: keeps sigmoid + `v_prediction`, adds
  `sigma_log_normal: [-0.4, 1.0]`.
- `minsnr5`: keeps sigmoid + `v_prediction`, adds `min_snr_gamma: 5.0`.

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
python scripts/prepare_nf_sweep_configs.py --project-dir "$PWD"
```

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
