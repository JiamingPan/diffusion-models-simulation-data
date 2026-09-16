# Runtime and validation

## Immutable inputs and pairing

`scripts/dit_sampler_matrix.py --prepare` freezes the selected checkpoints,
matching YAML checksums, raw model/config/shard checksums, conditions, task
coverage, original trace hashes, training-reference hash, and initial noise.
It verifies the original trace control's raw-weight, scheduler and config
contract. No substitute epoch or training subset is guessed.

Preparation produces `plan/plan.json` and `plan/initial_noise.npz`. The first
32 fields must exactly equal the previous saved trajectory inputs. CPU RNG
differences between runtimes fail closed. Each stochastic arm starts a separate
CPU step generator at seed 124, distinct from initial seed 123. Reusing seed 123
for two generators would repeat the first Gaussian field and correlate the first
solver increment with the initial noise; this is explicitly forbidden.
Step noise is shared across models within an arm; different timestep grids do
not constitute identical Brownian paths across different sampler arms.

Preparation freezes Python, Torch, NumPy, Diffusers and Hub versions. Tasks fail
closed if these change; completion records also include the actual scheduler
module checksum, device and GPU name so numerical differences are inspectable.

Each scheduler advances once for all 128 fields; only model inference is chunked.
This prevents multistep-history corruption and chunk-dependent random streams.
The SDE uses `DPMSolverMultistepScheduler` with
`algorithm_type="sde-dpmsolver++", solver_order=2`, not `DPMSolverSDEScheduler`.
No additional torchsde dependency is needed for this implementation.

## Great Lakes deployment (does not submit a job)

After the separately approved push, fetch the exact approved commit and create
a clean detached adapter worktree. Set `EXPECTED_COMMIT` to its full SHA and
`CODE_ROOT` to that worktree. Keep these runtime roots unchanged:

```bash
export PROJECT_DIR=/home/jiamingp/diffusion_models_repo
export RUNTIME_CODE_ROOT=/scratch/huterer_root/huterer0/jiamingp/iaifi_poster_code_555f350
export PYTHON_BIN=/home/jiamingp/venvs/cosmodiff_nf_class/bin/python
export COSMODIFF_PIN_ROOT=/scratch/huterer_root/huterer0/jiamingp/cosmodiff_seed_restart_pin_58c77eb_555f350
export COSMODIFF_PIN_MANIFEST="$COSMODIFF_PIN_ROOT/seed_restart_pin_manifest.json"
export EXPECTED_COSMODIFF_BASE_REVISION=58c77eb45de6e4d135697ba83ffee93ae54d918c
export MATRIX_OUT_DIR="$PROJECT_DIR/results/dit_l16_sampler_matrix_v1"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"
export COSMODIFF_DIR="$COSMODIFF_PIN_ROOT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$RUNTIME_CODE_ROOT"

# Existing CSV/galleries only: no inference or result writes.
"$PYTHON_BIN" "$CODE_ROOT/scripts/dit_sampler_matrix.py" \
  --inspect-trajectories --project-dir "$PROJECT_DIR"

# Run once; a pre-existing plan is never overwritten.
"$PYTHON_BIN" "$CODE_ROOT/scripts/dit_sampler_matrix.py" \
  --prepare --project-dir "$PROJECT_DIR" --out-dir "$MATRIX_OUT_DIR"
export MATRIX_PLAN_SHA256=$("$PYTHON_BIN" -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$MATRIX_OUT_DIR/plan/plan.json")
PREFLIGHT_ONLY=1 bash "$CODE_ROOT/scripts/slurm/sample_dit_sampler_matrix.sbatch"
```

The runtime verifier checks the **original** code root at
`555f350f82c913ff06150c96e66709719856cd83`; never repoint the immutable pin to
the adapter. The stub is first on PYTHONPATH, user site is disabled, and child
scoring keeps the interpreter/environment without `-S` or `-E`. There are no
Matplotlib imports. Native CPU smoke tests execute all four installed schedulers
through the cosine/zero-SNR endpoint and verify equal ODE/SDE 50-step grids.
This is scheduler compatibility verification, not real-weight model loading.

Preparation hashes model files using streamed reads. The preflight uses at most
two CPU threads and loads no model weights. It submits no job and performs no
GPU inference. Do not run actual sampling on a login node.

## Protected job submission

STOP after the successful preflight and review the actual frozen task list.
Preview the full Slurm command, adapter SHA, plan hash, output/log paths, account
`huterer2`, partition `spgpu`, and resolved array range. Obtain **APPROVE RUN**
for that exact command before submission.

There are 12 tasks for three models, or 16 if the selected L12 checkpoint is
available. Use one array at concurrency 2, one node/GPU per task, four CPUs,
80 GiB, and a one-hour walltime. The configured upper allocation is therefore
12 or 16 GPU-hours; actual usage may be lower. Queue delay is not a failure.
No new training, checkpoint copies, cancellation, generalization sweep, or
automatic paper-protocol changes are authorized by this matrix.

## Outputs and completion

Each `tasks/<model>__<condition>/` contains `samples.npz`, `summary.csv`,
`outcomes.csv`, and `complete.json`. The existing `evaluate_late_start.py` is
called in a real child process with explicit `--protocol sampler-matrix`.
Its default strict legacy late-start checks remain unchanged. Matrix mode
verifies the exact 256-map reference, config, steps, raw weights, RNG fields,
actual saved noise and per-sample/batch hashes.

No completion record is published before sampling **and scoring** succeed.
Directories are single-writer, exclusively locked, and atomically published.
Failed pending files remain available for diagnosis. Never overwrite completed
arms or remove stale locks without investigating and obtaining approval.

The DPM50 first-32 control reports output differences and outcome agreement
against the previous trace. Numerical differences are explicit qualifications,
not silently treated as a repeat of the old control.

After Slurm confirms all arms completed, this CPU-only aggregation submits no job:

```bash
"$PYTHON_BIN" "$CODE_ROOT/scripts/dit_sampler_matrix.py" \
  --summarize --out-dir "$MATRIX_OUT_DIR"
```

The aggregate fails closed on missing arms, changed checksums, changed task
coverage, mismatched condition/input metadata, reordered sample indices, noise
mismatch, or count/outcome disagreement. It writes `comparison/summary.csv`,
`paired_outcomes.csv`, `summary_original32.csv`, and a terminal checksummed JSON.
Expect roughly 200–300 MiB of samples/noise for the full matrix, not checkpoint
copies. Results use a new directory and do not overwrite earlier experiments.

## Interpretation

Read persistence alongside majority, local patch fit, global cosine gap,
boundary ratio, and galleries. Frozen/correlated noise can have high persistence;
low majority plus persistence near 1 is not unique proof of stitched copies.
The cohorts compare L8 on the actual L16-junk noise indices, not unmatched
aggregate populations. Training duration remains a confound for depth claims.

Reduced junk under DPM200 supports sensitivity to deterministic discretization.
Reduced junk under SDE50 versus ODE50 supports a stochastic-solver intervention
at equal step count. DDPM500 also changes solver and timestep grid. None of
these observations alone proves predictor sharpness, chaos, or excludes all
model/training/normalization interactions. Failure of this limited matrix also
does not prove that retraining is the only remedy.

Inspect both the original 32 and all 128 outcomes and normalized-field power
ratios below k=64. A low maximum cosine is not physical validity, and clean
training copies are not generalization. A zero observed junk count is not a
guarantee of zero failure probability. Choose any new paper-wide sampler or
generalization sweep only after reviewing these outputs and obtaining approval
for the expanded compute scope.
