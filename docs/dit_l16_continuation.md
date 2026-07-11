# DiT-L16 small-data continuation

This workflow tests whether the unusual DiT-L16 results at
`N_2D = 2^6, ..., 2^10` are caused by insufficient optimization.

## Controlled variables

- same five training sets and normalization
- same 16-layer DiT architecture
- same AdamW optimizer and cosine-restart learning-rate schedule
- same EMA settings, batch size, gradient accumulation, and null class label
- same 50-step DPM-Solver sampler and seed for evaluation

Only the number of optimizer updates changes.

## Stages and checkpoint behavior

The workflow adds four sequential stages of approximately 25,000 optimizer
updates, producing nominal 225k, 250k, 275k, and 300k checkpoints. Each stage
has an eight-hour walltime and writes recovery checkpoints approximately every
5,000 optimizer updates. A normally completed stage always writes its exact
stage-final checkpoint. If Slurm kills a stage at its walltime, rerun that same
stage; training resumes from the newest recovery checkpoint, so at most about
5,000 updates are repeated.

The manifest is frozen before submission. Do not regenerate it while the chain
is running, because all stages append checkpoints to the same run directories.

## Great Lakes

After pulling the current `main` branch:

```bash
cd /home/jiamingp/diffusion_models_repo

python scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py \
  --project-dir . --print-table

bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue.sh
```

The submitter limits each train/sample array to two concurrent GPU tasks and
waits for each stage's exact samples before beginning the next stage. It also
submits PCA and SSCD analysis jobs for each stage. Set `SUBMIT_ANALYSIS=0` to
submit only training and sampling.

Monitor with:

```bash
squeue -u jiamingp
sacct -j JOB_ID --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS
```

If one training stage times out, inspect the latest recovery checkpoint and
restart the chain from that stage without regenerating the manifest:

```bash
START_STAGE=2 REUSE_EXISTING_MANIFEST=1 \
  bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue.sh
```

Replace `2` with the interrupted stage. Runs that already reached that stage's
exact checkpoint exit immediately; unfinished runs resume from their newest
recovery checkpoint. The remaining stages are submitted behind the repaired
stage with fresh `afterok` dependencies.

## Real-reference provenance

The controlled UNet fidelity cell in
`notebooks/nf_generalize_fig2_partial_quickcheck.ipynb` now reloads the complete
training reference from the run config. For `N_2D = 2^10`, it must load exactly
1,024 slices or the cell raises `FULL TRAINING REFERENCE MISMATCH`.
