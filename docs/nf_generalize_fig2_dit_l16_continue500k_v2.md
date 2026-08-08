# DiT-L16 300k-to-500k Continuation Runbook

This workflow continues the ten clean `fresh300k_v2` DiT-L16 runs for training sizes `2^6` through `2^15`. It restores complete training state, samples every 40k updates, runs PCA/SSCD and physical diagnostics, and fails closed unless all expected artifacts pass the final audit.

## Start on Great Lakes

```bash
cd /home/jiamingp/diffusion_models_repo
bash scripts/gl_safe_pull.sh main
bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh
```

The submission creates five sequential training stages targeting 340k, 380k, 420k, 460k, and 500k updates. Each GPU array permits at most two active tasks. Training tasks request 48 hours and write recoverable checkpoints approximately every 5k updates.

## Monitor

```bash
squeue -u jiamingp

sacct -X \
  -j <precheck>,<training>,<sampling>,<pca>,<sscd>,<physics>,<audit> \
  --format=JobID%24,JobName%24,State,ExitCode,Elapsed,MaxRSS
```

Do not treat an empty `squeue` result as success. Confirm every job with `sacct`, then inspect:

```bash
cat local/nf_generalize_fig2_dit_l16_continue500k_v2/final_audit.json
```

The required terminal state is `"status": "PASS"`.

## Restart

If stage `S` fails, restart from that stage only after checking the error log:

```bash
START_STAGE=S REUSE_EXISTING_MANIFEST=1 \
  bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh
```

The restart path refuses to proceed when a required prior checkpoint, sample, or metric table is missing. Do not delete the frozen manifests between attempts.

## Notebook

After the final audit passes, rerun the existing results notebook:

```bash
cd /home/jiamingp/diffusion_models_repo
jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/nf_generalize_fig2_dit_results.ipynb
```

The tagged section `dit-l16-continue500k-v2` reads the final audit before plotting. It covers loss, PCA/SSCD novelty, exact-subset one-point and power-spectrum statistics, k-bin 20/40/60 variance, DPM50 versus DDPM500 controls, patch-boundary artifacts, and nearest-training matches.

## Storage

Large checkpoints and samples remain under scratch/results paths declared by the frozen manifests. Keep `/home` logs under observation because the home quota has previously exceeded 97%. Do not move active checkpoints while jobs or dependent analyses are running.
