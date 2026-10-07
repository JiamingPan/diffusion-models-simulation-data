# CLAUDE.md

Project: memorization vs generalization in diffusion models trained on CAMELS HI maps, and the
Omega_m recovery bias of conditional models at small training-set size. Paper in preparation
(JCAP/PRD) with Nick Kern. Owner: Jiaming Pan.

## How to work with me

- Be exact. Name files and commands. Show the command and its output path before asking me to run
  anything. I paste terminal output back.
- Never change the scope of a test silently. If you want a different test, say so and why.
- Do not overclaim. Every number in a summary must come from a result file; print the path. If a
  statement is not supported by a file, say "not supported by a result file".
- Short replies. Plain language. No bullet-point essays.
- When a result contradicts an earlier hypothesis, say so plainly.

## Repositories and where things live (Great Lakes)

- `/home/jiamingp/diffusion_models_repo`  : scripts, notebooks, configs, results. All sbatch jobs run
  from here. `results/` is here. This is the primary checkout; sync code with git, not rsync.
- `/home/jiamingp/Diffusion_model/cosmo_diffusion_main` : the `cosmodiff` training package
  (`cosmodiff/utils.py`, `optim.py`, `augment.py`, `transform.py`, `scripts/cosmodiff_train.py`).
  Training imports from here even though results land in `diffusion_models_repo`.
- Mac checkout: `/Users/apple/AI/Diffusion_model` (same GitHub repo, JiamingPan/diffusion-models-simulation-data).
  Pull with git. rsync only for git-ignored files (figures, samples), from the Mac terminal, never inside ssh.
- Checkpoints, samples, prepared data: `/scratch/huterer_root/huterer0/jiamingp/saved_runs/...`
- DiT plans: `/scratch/huterer_root/huterer0/jiamingp/{dit_l16_adalnzero_highn_screen_v1,dit_zero_remaining_300k_v1}/plan.json`
- CAMELS grids: `/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/Grids_HI_IllustrisTNG_{LH,CV}_128_z={0.0,1.0,2.0}.npy`
- VGG probe weights cache: `/nfs/turbo/lsa-huterer/jiamingp/torch_cache`
- Old logs: `/scratch/huterer_root/huterer0/jiamingp/logs_archive` (home quota was 95%).

## Environment rules (hard)

- venv `cosmodiff_nf_class`. Do NOT touch `00-cosmodiff-base-venv.pth` or include-system-site-packages.
- No pip installs, no downloads, no env changes.
- Evaluate/report jobs need
  `export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:$LD_LIBRARY_PATH` (in the sbatch templates).
- Interactive: `PYTHONPATH="$STUB_ROOT:$COSMODIFF_DIR:$PROJECT_DIR:$PROJECT_DIR/scripts"`;
  for probe steps `PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/scripts"`.
  `PROJECT_DIR=/home/jiamingp/diffusion_models_repo`, `COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main`.
- Slurm account: **huterer2** for every project job (`#SBATCH -A huterer2`). Never huterer0 (the `/scratch/huterer_root/huterer0/`
  paths are storage only) and never cavestru0 (another project). Check the account in every sbatch file and submit command.
- Slurm: array limit `%2`. Attach to a running allocation with `srun --jobid=<id> --overlap --pty bash`
  rather than queueing. Check jobs with `sacct -X -j <id> --format=JobID%20,JobName%40,State,Elapsed,WorkDir%80`.
- Never `--overwrite` existing samples or results. New outputs get new paths or new `sample_label`s.
- Do not submit jobs without showing me the sbatch file and the submit command first.
- Tests: `tests/` (18 pass on Mac and GL). Run them after any script change.

## Training recipe (verified from configs, 2026-10-01)

- Unconditional sweeps: `UNet2DModel` (u64/u128/u256 = top channel width 64/128/256) and DiT
  (L8/L12/L16, patch 8, adaLN-Zero). Conditional: `UNet2DConditionModel`, cross-attention on a single
  6-parameter token (`encoder_hid_dim: 6`, `cross_attention_dim: 32`).
- `DDPMScheduler`, `num_train_timesteps: 500`, `squaredcos_cap_v2`, `rescale_betas_zero_snr: true`,
  `prediction_type: v_prediction`. `min_snr_gamma: 5.0`. `sigma_log_normal: null` (uniform timesteps).
  EMA tracked (`ema_sigma_rels [0.02, 0.1]`) but NOT used at sampling. No guidance. Batch 32, AdamW 1e-4,
  wd 0.01, CosineAnnealingWarmRestarts T_0 4000.
- Data transform: `log`, then center-max, then asymmetric tanh (alpha 0.8, beta 10, sigma 1.5).
  Unconditional runs fit `center`/`xmax` from each run's own training subset (per-run normalization).
  Conditional runs fix them (center 7.896381651151516, xmax 22.404052734375).
- Unconditional data: LH+CV at z = 0, 1, 2, `zthin: 8` => 16 slices per simulation. N counts slices,
  so N = 64 is 4 simulations. Slices from one box are correlated. Remember this in any i.i.d. argument.
- NO augmentation in any existing run. Verified at runtime (eight fetches identical, no
  `augmentations.pkl` in checkpoints). `cosmodiff/augment.py` has `RandomDihedral2D` ready for D4 runs.
- Budgets: UNet sweep 200k updates, DiT sweep 300k, conditional runs 200k.
- Sampling: `DPMSolverMultistepScheduler`, 50 steps, seed 123, single Generator, deterministic.
  `sample_label` for the unconditional sweeps is `dpm50_n512`. Unconditional sampler settings are set by
  the sampling script, not the yaml (yaml has `generate.scheduler: null`); confirm them before comparing.

## Sweeps and runs

- `nf_conditional_bias_fresh_full_sweep_200k`: conditional, N = 64 ... 32768, one map per cosmology.
- `nf_conditional_fixedC64_multiplicity_200k`: 64 fixed cosmologies x m = 1, 2, 4, 8 slices
  (runs `nf_cond_fixedC64_m{01,02,04,08}_n{64,128,256,512}_fresh200k`). DONE, see results below.
- `nf_conditional_omsig_continuous_200k`: 2-d (Omega_m, sigma_8) label. Jobs 62232389 -> 62232390 -> 62232391.
- `nf_conditional_omsig_class36_200k`: `UNet2DModel` with 37 class embeds, 6x6 bins
  Omega_m in linspace(0.1,0.5,7), sigma_8 in linspace(0.6,1.0,7), class = 6*iOm + iSig.
  Jobs 62232393 -> 62232394 -> 62232395. m per class is about N/36.
- Unconditional: `local/nf_generalize_fig2/manifest.json` (UNet-64/128/256, N = 64 ... 32768, 200k);
  DiT L8/L12/L16 plans above (300k). PCA95 notebook:
  `notebooks/...dit_l16_zero_init_generalization_pca95` (shared 32-component PCA, real-real NN 95th pct
  threshold, filters `num_layers == 16 and patch_size == 8` so it currently drops UNet entries).

## What is established (do not go beyond this)

- Conditional N = 64 is a theta -> image lookup: at its 64 training cosmologies it reproduces the training
  map (centered pixel cosine 0.998, draw-to-draw 0.996, paired probe shift -0.004).
  Files: `results/nf_conditional_bias_fresh_full_sweep_200k/samples_theta_interp/*_train_theta_report.json`,
  `_per_cosmology.csv`, `train_theta_gallery.png`.
- At held-out theta it produces one degraded image per theta (cosine 0.67 to nearest training map,
  NNLS top-k R^2 0.49 vs 0.014 real, half the power, bias -0.154). Blending real maps reproduces the sign.
  Sparse coverage does not set the sign. Supported wording: "lookup-like memorization; one degraded image
  per unseen theta". NOT supported: "linear superposition".
- Memorization criterion follows Gu et al. 2023 / arXiv:2505.20123: generate at the training theta and
  compare to the training map. `simdiff_eval/train_theta_report.py` compares to the stored map only (no D4
  orbit search). Add `--orbit d4` as an option before scoring any augmented run; default unchanged.
- Multiplicity result (files under `results/nf_conditional_fixedC64_multiplicity_200k/`):
  training theta draw-to-draw cosine 0.996, 0.479, 0.195, 0.186 for m = 1, 2, 4, 8; nearest-training cosine
  0.998, 0.995, 0.988, 0.962 (still copying, from a bigger set). Held-out median Omega_m bias -0.123, -0.113,
  -0.145, -0.123; 68% coverage 1, 4, 3, 6 of 32. Full sweep at equal N: N = 256 bias -0.108 (1/32),
  N = 512 bias -0.074 (5/32). Defensible wording: "adding cosmologies and adding realizations are not
  interchangeable for median bias; whether local parameter-space coverage explains it is untested".
  One seed. Do not re-index figures as if the mechanism were established; show both axes.
- Unconditional transition (PCA95 score, saved samples): DiT-L16 crosses 0.5 between N = 1024 and 2048;
  UNet-64 by N = 256, UNet-128/256 near 1024. UNet points were scored with a different PCA setup
  ("PCA6697") and budget (200k vs 300k); not yet comparable on one axis. L16 at N = 256 missing.

## Theory frame (Hunt, Kamb, Ganguli, arXiv:2607.08041)

A Bayesian denoiser with observation C_t memorizes when I(phi; C_t) > ln(number of training items it
chooses among), generalizes otherwise. Gaussian upper bound I <= 0.5 Tr ln(I + snr_t Sigma) =
0.5 sum_k ln(1 + snr_t P(k)) for a stationary field, computed in the model's normalized pixel space.
Full-image I exceeds ln N at low noise for every N we train, so the observed transition reflects the
network's information restriction (effective patch scale), not full-image lookup. For conditional models
the label is part of the observation; the count at a held-out theta is unknown and is what the
multiplicity and class36 runs probe. This is a frame for predictions, not an established result.

## Agent loop (read docs/agent_loop.md)

- Three agents in `.claude/agents/`: planner (proposes a run with a prediction, writes a `planned`
  ledger entry), executor (prepares files, runs CPU steps, stops at the gate for GPU/submit/delete),
  critic (checks every number in a claim against a file; rewrites to the defensible version).
- `experiments/ledger.jsonl` is append-only and written only through `scripts/ledger.py`
  (plan / submit / result / fail / show). Every quoted number traces to a `done` entry.
- `.claude/hooks/guard_sbatch.py` blocks `sbatch` unless `RUN_NAME=<run>` is in the command and the
  run has a `planned` entry (`RUN_NAME=analysis` is allowed for CPU jobs), and blocks rm, scancel,
  pip, conda, --overwrite, git push.
- Approval words: APPROVE RUN, APPROVE EDIT, APPROVE PUSH, APPROVE SEND, APPROVE DELETE. Show the exact
  command, cost and side effects first. An implied yes is not approval.
- Metrics come from `scripts/evaluate_run.py` only. No new metrics inside a run.

## Open items, in order

1. Posterior entropy (model-free, CPU). `notebooks/posterior_entropy_memorization.ipynb`, provenance
   pattern of the PCA95 notebook. For each verified DiT-L16 run: exact training subset, that run's fitted
   log/center-max/tanh normalization (rebuild via `parse_config_data`), 256 held-out real maps, 25 values of
   sigma_t^2 in [1e-3, 1e2] from the run's 500-step scheduler, softmax posterior
   log w_i = -||phi_t - sqrt(abar) phi_i||^2 / (2(1-abar)), entropy S, deficit ln N - S, Gaussian bound from
   mean 2-D P(k) of normalized real maps, patch versions L in {5, 9, 17, 33, 65, 128}. Outputs under
   `notebooks/executed/posterior_entropy/` (summary.csv, provenance.json, figure). Report sigma_t^2 where
   S < 1 nat per N.
2. Conditional posterior entropy: same, posterior restricted to maps sharing the label (multiplicity m,
   class36 bins). `notebooks/executed/posterior_entropy_conditional/`.
3. Network observation scale: DiT-L16 N = 64, 1024, 4096 and UNet-256 N = 64, 1024 checkpoints; perturb one
   pixel of phi_t, radius containing 90% of |delta x0-prediction| per sigma_t; overlay L_c(sigma_t; N) from
   item 1. Inference only.
4. Multiplicity figure: `scripts/plot_nf_conditional_multiplicity_figure.py` ->
   `results/nf_conditional_fixedC64_multiplicity_200k/figures/multiplicity_summary.{png,pdf,csv}`.
   Panels vs m: draw-to-draw cosine (held-out solid, training dashed), |median bias| with bootstrap band,
   68% coverage; full-sweep N = 64..512 as a separate marker series. Every value from a result file.
5. When 62232391 / 62232395 finish: 1-1 recovery and coverage per N for both variants (class36 against
   bin centres with bin width shown), draw-to-draw cosine, comparison at equal N with the 6-parameter sweep.
6. Re-score UNet samples with the PCA95 notebook (PCA32, q = 0.95) so UNet and DiT share one definition.
   Recipe diff UNet vs DiT yaml (dataset, min_snr, EMA, budget, sampler) before any UNet rerun.
7. Fixed-cosmology extension: 64 cosmologies x 64 and x 128 slices (N = 4096, 8192). Prepare only.
8. D4 augmentation: `augmentations: {RandomDihedral2D: {dims: [-2, -1], p: 1.0}}`, suffix `_d4aug`,
   unconditional u256 and dit_l16 at N = 64..1024, conditional m01 at N = 64 and 512; plus continue the
   memorized N = 64 checkpoint with D4 on, checked at several checkpoints; plus inference-time D4
   symmetrization of a memorized checkpoint (`sample_label dpm50_n512_d4sym`). Score with `--orbit d4`.
   Prepare only; run after item 1 gives the I curve. Prediction on record: transition moves down by at
   most a factor 8 in N; N = 64 stays memorized.
9. Sampler robustness (prepare only): dpm++ 50/100, sde-dpm++ 50, karras sigmas, final_sigmas_type;
   for DiT-L16 N = 1024, 2048 and u256 N = 512, 1024; new sample_labels.
10. Paper: Criterion 1 with the conditional memorization test, training-theta and held-out galleries,
    both axes (cosmologies, maps per cosmology) in conditional figures, Test 4 table.

Not doing: stretch/warp augmentation (changes P(k), breaks the theta mapping). Equivariant UNet only if
item 8 shows the shift (design note first, no code).
