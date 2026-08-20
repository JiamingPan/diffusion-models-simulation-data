# Conditional calibration transition sweep

This sweep measures how conditional response changes with training-set size,
rather than comparing only two hand-picked endpoints.

## Controlled protocol

- Train ten fresh conditional UNet-128 generators at
  `N_2D = 2^6, ..., 2^15`.
- Give every generator the same target of 200,000 optimizer updates.
- Use deterministic nested training subsets and training seed 123.
- Condition on the complete six-dimensional CAMELS parameter vector.
- Reserve CAMELS simulations 900-931 for evaluation.
- Evaluate every generator with the same frozen VGG16+MLP probe.
- Do not reuse the earlier `2^7` or `2^14` generator checkpoints.

## Great Lakes entry point

```bash
cd /home/jiamingp/diffusion_models_repo
bash scripts/slurm/submit_nf_conditional_bias_full_sweep.sh
```

The submission chain prepares data, checks the protocol, trains and samples
all ten sizes with at most two one-GPU tasks active, evaluates the frozen
probe, and audits the outputs.

## Result files

The evaluation writes these figures under
`results/nf_conditional_bias_fresh_full_sweep_200k/calibration_vgg/`:

- `bias_probe_omega_m_transition_vs_dataset_size.png`: all fitted
  Omega-m response lines and the response slope versus training-set size.
- `bias_probe_omega_m_all_dataset_sizes.png`: one calibration panel for each
  of the ten training-set sizes.
- `bias_probe_all_parameter_slopes_vs_dataset_size.png`: response slopes for
  all six conditioning parameters.

The figures are also displayed by
`notebooks/nf_conditional_bias_vgg_results.ipynb` after the pipeline finishes.
