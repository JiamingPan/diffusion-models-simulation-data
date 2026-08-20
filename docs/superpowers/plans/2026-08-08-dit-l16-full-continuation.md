# DiT-L16 Full-Sweep Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the ten clean DiT-L16 300k runs to 500k optimizer updates with full-state resume, evaluate matched checkpoints, separate sampler failures from training failures, and integrate the resulting novelty and physical-statistics diagnostics into the existing DiT results notebook.

**Architecture:** Add an isolated `nf_generalize_fig2_dit_l16_continue500k_v2` workflow that reads the frozen fresh-300k manifest, derives immutable 340k through 500k targets, and runs exact-target training, sampling, PCA/SSCD analysis, physical diagnostics, and a final audit through explicit Slurm dependencies. Reuse the existing class-safe exact-target resume adapter and exact training-subset loaders. Extend sample provenance with scheduler-terminal metadata, keep diagnostics in a new `simdiff_eval.dit_diagnostics` module, and update the existing notebook through an idempotent metadata-tagged cell block.

**Tech Stack:** Python 3.10, PyTorch, Diffusers, NumPy, pandas, SciPy, PyYAML, nbformat, Matplotlib, Slurm, unittest/pytest.

## Global Constraints

- Treat `nf_generalize_fig2_dit_l16_fresh300k_v2` as the only valid source sweep. Do not read or resume the legacy 200k or state-reset continuation directories.
- Continue every DiT-L16 dataset size from `d2p06` through `d2p15`; do not select checkpoints or datasets according to the desired scaling relation.
- Preserve model, EMA, optimizer, learning-rate scheduler, scaler, and RNG state through the existing class-safe resume path.
- Preserve architecture and preprocessing. The continuation precheck must prove image size 128, patch size 8, a 16 by 16 token grid, 12 attention heads, head dimension 64, 32 normalization groups, constant class label 0, and no augmentation.
- Use exact targets at 300k, 340k, 380k, 420k, 460k, and 500k updates. The 40k spacing is phase-matched to the 4k cosine-restart period.
- Allow at most two training or sampling GPU array tasks at once. Give each training task 48 hours and retain recovery checkpoints approximately every 5k updates.
- Sample 512 maps with DPM-Solver, 50 steps, and seed 123 for every dataset and target. Run DDPM-500 controls only for `d2p08` and `d2p11` at 300k and 500k.
- Every physical reference must come from the exact configured training subset for that model. Never substitute the complete CAMELS collection.
- Modify the dirty notebook and dirty tests additively. Do not discard existing outputs or user edits.
- Do not commit unrelated dirty files. Stage exact paths for every task commit.

---

## File Map

**New continuation workflow**

- `scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py`: derive immutable continuation rows and YAMLs from the frozen fresh-300k source manifest.
- `scripts/check_nf_generalize_fig2_dit_l16_continue500k_v2.py`: validate source checkpoints, exact targets, full resume state, config parity, and architecture/preprocessing parity.
- `scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py`: verify all expected checkpoints, samples, provenance fields, and analysis tables.
- `scripts/slurm/precheck_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch`: run config, checkpoint, and preflight sampling checks.
- `scripts/slurm/train_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch`: resume one exact stage for all ten datasets.
- `scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch`: produce DPM-50 samples for one stage.
- `scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_ddpm_controls.sbatch`: produce the four DDPM-500 controls.
- `scripts/slurm/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.sbatch`: compute one-point, power-spectrum, selected-bin variance, and patch-boundary tables.
- `scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch`: run the final artifact audit.
- `scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh`: submit the dependency chain and support stage restart.

**Sampling and diagnostics**

- `scripts/sample_cosmodiff.py`: record the actual inference timestep sequence and terminal scheduler state in each `.npz`.
- `simdiff_eval/dit_diagnostics.py`: pure, tested functions for bootstrap intervals, selected-bin power statistics, common-bin one-point error, and patch-boundary artifacts.
- `scripts/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.py`: load exact references and samples, call the diagnostic functions, and write tidy tables.

**Notebook integration**

- `scripts/update_dit_continue500k_v2_notebook.py`: idempotently replace a tagged continuation section in the existing notebook.
- `notebooks/nf_generalize_fig2_dit_results.ipynb`: show all-ten loss trajectories, full novelty curves, full one-point and power-spectrum sweeps, selected-bin variance, sampler controls, nearest-training examples, and patch diagnostics.

**Tests and documentation**

- `tests/test_nf_fig2_dit_l16_continue500k_v2.py`: manifest arithmetic, config parity, source isolation, Slurm dependencies, and result-audit tests.
- `tests/test_dit_l16_diagnostics.py`: numerical unit tests for the diagnostic functions.
- `tests/test_nf_fig2_continuation_guards.py`: sampling-provenance and terminal-scheduler regression tests.
- `tests/test_dit_results_notebook_presentation.py`: notebook section, labels, all-ten coverage, and idempotence checks.
- `scripts/slurm/README.md`: operator commands, restart semantics, outputs, and audit instructions.

---

### Task 1: Freeze the 300k-to-500k continuation manifest

**Files:**
- Create: `scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py`
- Create: `tests/test_nf_fig2_dit_l16_continue500k_v2.py`

- [ ] **Step 1: Add failing source-manifest and stage-arithmetic tests**

Create a temporary fresh-300k manifest with ten rows and minimal source YAMLs. Test the public functions:

```python
SOURCE_SWEEP_NAME = "nf_generalize_fig2_dit_l16_fresh300k_v2"
CONTINUE_SWEEP_NAME = "nf_generalize_fig2_dit_l16_continue500k_v2"
TARGET_UPDATES = (340_000, 380_000, 420_000, 460_000, 500_000)

def build_continuation_rows(
    project_dir: Path,
    source_rows: list[dict[str, Any]],
    *,
    target_updates: tuple[int, ...] = TARGET_UPDATES,
) -> list[dict[str, Any]]: ...
```

Assert:

- ten source rows become fifty continuation rows;
- every dataset tag `d2p06` through `d2p15` occurs once per target;
- every row names the exact 300k source checkpoint and exact previous/target checkpoint;
- target epochs use `ceil(target_updates / optimizer_steps_per_epoch) - 1` consistently;
- sample labels are `dpm50_cont_340k` through `dpm50_cont_500k`;
- source and continuation checkpoint roots are different;
- 40k is divisible by the 4k restart period;
- duplicate or missing dataset tags fail.

- [ ] **Step 2: Run the new test and confirm it fails because the preparation module is absent**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
```

Expected: import failure for `prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs`.

- [ ] **Step 3: Implement immutable row generation and YAML cloning**

The preparation script must:

1. Read `local/nf_generalize_fig2_dit_l16_fresh300k_v2/manifest.json`.
2. Require exactly ten completed source rows with tags `d2p06` through `d2p15` and `target_updates == 300000`.
3. Read each source YAML and copy it without rebuilding model defaults.
4. Change only `io.output_dir`, `train.num_epochs`, and `train.checkpoint_every_n_epochs`.
5. Compute a SHA-256 digest of the source YAML and store it in every continuation row.
6. Write `manifest.json`, `analysis_manifest.json`, and stage-specific YAMLs under `local/nf_generalize_fig2_dit_l16_continue500k_v2/`.
7. Refuse to overwrite a non-identical frozen manifest when `--use-existing-manifest` is set.
8. Include the ten frozen 300k source samples in `analysis_manifest.json` so the baseline is provenance-checked rather than reconstructed from a legacy run.
9. Expose `--check-only` and `--seed-checkpoints` for later Slurm prechecks.

Store these provenance fields in every row:

```python
{
    "source_sweep_name": SOURCE_SWEEP_NAME,
    "source_config": str(source_config),
    "source_config_sha256": sha256_file(source_config),
    "source_checkpoint": str(source_checkpoint_300k),
    "previous_expected_checkpoint": str(previous_checkpoint),
    "expected_checkpoint": str(target_checkpoint),
    "target_total_updates": target_updates,
    "restart_period_updates": 4_000,
    "checkpoint_every_target_updates": 5_000,
    "sample_label": f"dpm50_cont_{target_updates // 1000}k",
}
```

- [ ] **Step 4: Add strict config-equivalence assertions**

Implement:

```python
def assert_continuation_config(
    source_path: Path,
    continuation_path: Path,
    row: dict[str, Any],
) -> None: ...
```

Recursively compare the source and continuation YAML after removing exactly the three allowed mutable keys. Fail with the full dotted paths of any other differences.

- [ ] **Step 5: Seed isolated continuation directories from the exact 300k checkpoints**

Implement:

```python
def seed_continuation_directories(
    rows: list[dict[str, Any]],
    *,
    copy_function: Callable[[Path, Path], None] = copy_checkpoint_tree,
) -> None: ...
```

For each dataset, copy the complete exact 300k checkpoint tree into its new continuation checkpoint directory. Verify the source and destination inventories, file sizes, and state-file hashes. If the destination exists, accept it only when it is byte-identical; otherwise fail rather than merge or overwrite it. The original fresh300k-v2 directory remains read-only.

- [ ] **Step 6: Run the focused tests**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
```

Expected: all manifest, arithmetic, immutability, and config-equivalence tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py tests/test_nf_fig2_dit_l16_continue500k_v2.py
git commit -m "Prepare full DiT-L16 continuation sweep"
```

---

### Task 2: Prove checkpoint and architecture parity before using GPUs

**Files:**
- Create: `scripts/check_nf_generalize_fig2_dit_l16_continue500k_v2.py`
- Create: `scripts/slurm/precheck_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch`
- Modify: `tests/test_nf_fig2_dit_l16_continue500k_v2.py`

- [ ] **Step 1: Add failing tests for the required precheck assertions**

Test pure validation helpers using temporary checkpoint directories and YAML files:

```python
EXPECTED_DIT_L16 = {
    "sample_size": 128,
    "patch_size": 8,
    "num_layers": 16,
    "num_attention_heads": 12,
    "attention_head_dim": 64,
    "norm_num_groups": 32,
}

def validate_source_row(row: dict[str, Any], project_dir: Path) -> dict[str, Any]: ...
def compare_architecture_configs(config_paths: dict[str, Path]) -> dict[str, Any]: ...
```

Assert failure for:

- missing exact 300k checkpoint;
- missing optimizer, LR scheduler, scaler, EMA, or RNG state;
- source config digest mismatch;
- patch size not equal to 8;
- L8/L12/L16 differences outside `num_layers` and run/output fields;
- any augmentation or nonzero class label.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
```

Expected: missing validation module or missing validation helpers.

- [ ] **Step 3: Implement the precheck CLI**

The CLI must:

1. Load and revalidate the frozen continuation manifest.
2. Call the existing `scripts/check_nf_generalize_fig2_dit_resume.py` logic for all ten source checkpoints.
3. Inspect checkpoint `config.json`, `checkpoint_config.yaml`, optimizer state, scaler state, random states, EMA state, and scheduler state.
4. Compare L8, L12, and L16 source configs and print a table that proves identical patch/preprocessing settings with depth as the intended architecture difference.
5. Write `local/nf_generalize_fig2_dit_l16_continue500k_v2/precheck_report.json` only after every row passes.
6. Exit nonzero on the first incomplete or ambiguous source.

- [ ] **Step 4: Implement the Slurm precheck**

The precheck job uses one GPU only for model reconstruction and sampler preflight. It must:

- create the log directory before submission;
- run config/checkpoint validation;
- run `scripts/sample_cosmodiff.py --preflight-only` on the smallest and largest source checkpoints;
- require the precheck report before exiting successfully.

- [ ] **Step 5: Run tests and shell syntax checks**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
bash -n scripts/slurm/precheck_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch
```

Expected: all tests pass; shell syntax check is silent.

- [ ] **Step 6: Commit Task 2**

```bash
git add scripts/check_nf_generalize_fig2_dit_l16_continue500k_v2.py scripts/slurm/precheck_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch tests/test_nf_fig2_dit_l16_continue500k_v2.py
git commit -m "Guard DiT-L16 full-state continuation"
```

---

### Task 3: Build exact-target training and restart-safe submission

**Files:**
- Create: `scripts/slurm/train_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch`
- Create: `scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh`
- Modify: `tests/test_nf_fig2_dit_l16_continue500k_v2.py`

- [ ] **Step 1: Add failing submission-contract tests**

Assert the training and submission scripts contain:

- `#SBATCH --time=48:00:00`;
- `#SBATCH --array=0-9%2` or submission-time `--array=0-9%2`;
- `scripts/run_cosmodiff_train_with_dit_resume.py`;
- both `--minimum-checkpoint` and `--target-checkpoint`;
- five sequential stages, each dependent on the previous stage's successful sample job;
- `START_STAGE` validation from 1 through 5;
- `REUSE_EXISTING_MANIFEST=1` required for restarts after stage 1;
- no reference to the legacy continuation sweep;
- no `afterany` dependency for scientific outputs.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
```

Expected: assertions fail because the Slurm files do not exist.

- [ ] **Step 3: Implement the training array**

For `CONTINUE_STAGE` 1 through 5, select exactly one row per array index. The job must:

1. Require the precheck report.
2. Activate the existing `cosmodiff_nf` environment.
3. Apply the current cosmodiff compatibility patches.
4. Run the class-safe resume checker immediately before training.
5. Invoke `run_cosmodiff_train_with_dit_resume.py` with the row's exact previous and target checkpoints.
6. Verify the exact target checkpoint exists after training.
7. Write a per-task completion JSON containing source checkpoint, target checkpoint, elapsed time, final epoch, and config digest.

- [ ] **Step 4: Implement the submission chain**

Submit:

1. freeze or reuse the manifest and seed byte-verified 300k checkpoints into isolated continuation directories;
2. precheck;
3. stage 1 train and DPM sampling;
4. stage 1 PCA, SSCD, and physics analysis;
5. stages 2 through 5 with the same dependency structure;
6. final DDPM controls;
7. final audit.

Use arrays limited to `%2`. Print every job ID and a compact restart command. Reject `START_STAGE > 1` unless the frozen manifest and prior exact checkpoint set both pass `--check-only`.

- [ ] **Step 5: Run tests and shell syntax checks**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
bash -n scripts/slurm/train_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch
bash -n scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh
```

Expected: tests pass; shell syntax checks are silent.

- [ ] **Step 6: Commit Task 3**

```bash
git add scripts/slurm/train_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh tests/test_nf_fig2_dit_l16_continue500k_v2.py
git commit -m "Submit staged DiT-L16 continuation"
```

---

### Task 4: Make sampler completion auditable

**Files:**
- Modify: `scripts/sample_cosmodiff.py`
- Create: `scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch`
- Create: `scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_ddpm_controls.sbatch`
- Modify: `tests/test_nf_fig2_continuation_guards.py`
- Modify: `tests/test_nf_fig2_dit_l16_continue500k_v2.py`

- [ ] **Step 1: Add failing scheduler-audit tests**

Add a fake scheduler with known `timesteps` and `sigmas`. Require:

```python
def scheduler_audit_metadata(scheduler, requested_steps: int) -> dict[str, Any]: ...
```

The returned and saved metadata must include:

- `requested_inference_steps`;
- `executed_inference_steps`;
- `first_timestep`;
- `final_timestep`;
- `terminal_sigma`, when exposed by the scheduler;
- `terminal_sigma_is_zero`, when exposed by the scheduler;
- the scheduler class name.

Do not infer zero terminal noise merely from the final timestep. If the scheduler does not expose terminal sigma, save `NaN` and `False` for the verifiability flag.

- [ ] **Step 2: Run the sampler tests and confirm failure**

Run:

```bash
python -m pytest tests/test_nf_fig2_continuation_guards.py -q
```

Expected: missing helper or missing `.npz` provenance keys.

- [ ] **Step 3: Implement scheduler provenance without changing generated tensors**

Compute the audit after `set_timesteps` and before sampling. Pass it to `save_sample_output` through an optional keyword so older callers remain valid. Save scalar values in the `.npz` beside existing checkpoint, config, scheduler, step, and seed provenance.

- [ ] **Step 4: Implement DPM-50 and DDPM-500 sampling arrays**

The DPM array must sample all ten rows at every target with:

```text
DPMSolverMultistepScheduler, 50 steps, 512 samples, batch size 8, seed 123
```

The DDPM control array must sample only:

```text
d2p08 at 300k, d2p08 at 500k, d2p11 at 300k, d2p11 at 500k
```

Use 500 steps, 512 samples, batch size 8, and seed 123. Name files and labels so DPM and DDPM cannot overwrite each other.

- [ ] **Step 5: Add sample guards**

After each sample file is written, verify:

- shape is `(512, 1, 128, 128)`;
- all values are finite;
- resolved checkpoint equals the requested exact target;
- executed steps equal requested steps;
- scheduler metadata is present;
- repeated sample labels do not point to byte-identical files from different checkpoints.

- [ ] **Step 6: Run focused tests and shell syntax checks**

Run:

```bash
python -m pytest tests/test_nf_fig2_continuation_guards.py tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
bash -n scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch
bash -n scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_ddpm_controls.sbatch
```

Expected: all tests pass; shell syntax checks are silent.

- [ ] **Step 7: Commit Task 4**

```bash
git add scripts/sample_cosmodiff.py scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_ddpm_controls.sbatch tests/test_nf_fig2_continuation_guards.py tests/test_nf_fig2_dit_l16_continue500k_v2.py
git commit -m "Audit DiT inference schedules"
```

---

### Task 5: Add physical and patch diagnostics

**Files:**
- Create: `simdiff_eval/dit_diagnostics.py`
- Create: `scripts/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.py`
- Create: `scripts/slurm/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.sbatch`
- Create: `tests/test_dit_l16_diagnostics.py`
- Modify: `tests/test_nf_fig2_dit_l16_continue500k_v2.py`

- [ ] **Step 1: Add failing numerical tests**

Test these public functions:

```python
def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 123,
) -> tuple[float, float]: ...

def selected_power_bin_statistics(
    power_spectra: np.ndarray,
    bin_indices: tuple[int, ...] = (20, 40, 60),
    *,
    n_resamples: int = 2_000,
    seed: int = 123,
) -> list[dict[str, float]]: ...

def one_point_l1_common_bins(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    bins: int = 120,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> float: ...

def patch_boundary_statistics(
    images: np.ndarray,
    *,
    patch_size: int = 8,
) -> dict[str, float]: ...
```

Use synthetic inputs to prove:

- constant selected-bin values have zero variance and collapsed intervals;
- bootstrap output is deterministic under a fixed seed;
- identical fields have zero one-point L1;
- smooth fields have a patch-boundary ratio near one;
- artificial 8-pixel tile seams produce a ratio significantly above one;
- invalid bin indices and wrong tensor shapes fail clearly.

- [ ] **Step 2: Run the diagnostic tests and confirm failure**

Run:

```bash
python -m pytest tests/test_dit_l16_diagnostics.py -q
```

Expected: import failure because `simdiff_eval.dit_diagnostics` does not exist.

- [ ] **Step 3: Implement the pure diagnostic functions**

Keep this module independent of plotting and file I/O. For patch boundaries, compare absolute neighbor differences crossing every eighth row/column with equal-size off-boundary neighbor differences and report mean, median, ratio, and excess.

- [ ] **Step 4: Implement the physical-analysis CLI**

For each sample row:

1. Load the sample `.npz` and validate provenance.
2. Stream the exact training subset with `iter_real_reference_batches_from_config`.
3. Compute a shared-bin one-point reference and L1 error.
4. Compute generated per-sample power spectra with `batch_power_spectra(..., nbins=91)`.
5. Compute the exact-reference mean power spectrum in bounded batches.
6. Write low-, mid-, and high-k errors plus per-sample variance, standard deviation, and bootstrap intervals at bins 20, 40, and 60.
7. Compute patch-boundary metrics for generated and real fields.
8. Accept the original DiT analysis manifest as `--baseline-manifest` and add 200k L8 and L12 patch-boundary rows for all ten dataset sizes, clearly labeled as baseline architectures with a different update budget.
9. Write tidy CSV tables under `results/nf_generalize_fig2_dit/tables/` and compact arrays under `results/nf_generalize_fig2_dit/physics/`.

Required output names:

```text
nf_generalize_fig2_dit_l16_continue500k_v2_physics_summary.csv
nf_generalize_fig2_dit_l16_continue500k_v2_pk_selected_bins.csv
nf_generalize_fig2_dit_l16_continue500k_v2_patch_boundaries.csv
nf_generalize_fig2_dit_l16_continue500k_v2_curves.npz
```

- [ ] **Step 5: Implement the Slurm analysis wrapper**

Use a standard CPU node, 8 CPUs, 80 GB RAM, and enough walltime for all exact references. Accept `SAMPLE_LABEL` and `OUT_PREFIX`; fail on missing samples instead of silently skipping a dataset.

- [ ] **Step 6: Run tests and CLI smoke checks**

Run:

```bash
python -m pytest tests/test_dit_l16_diagnostics.py tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
python scripts/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.py --help
bash -n scripts/slurm/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.sbatch
```

Expected: tests pass, help text prints, and shell syntax check is silent.

- [ ] **Step 7: Commit Task 5**

```bash
git add simdiff_eval/dit_diagnostics.py scripts/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.py scripts/slurm/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.sbatch tests/test_dit_l16_diagnostics.py tests/test_nf_fig2_dit_l16_continue500k_v2.py
git commit -m "Add DiT physical transition diagnostics"
```

---

### Task 6: Add PCA, SSCD, and final artifact audits

**Files:**
- Create: `scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py`
- Create: `scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch`
- Modify: `scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh`
- Modify: `tests/test_nf_fig2_dit_l16_continue500k_v2.py`

- [ ] **Step 1: Add failing audit tests with a temporary result tree**

Require:

```python
def audit_results(project_dir: Path, manifest_path: Path) -> dict[str, Any]: ...
```

The test tree must fail when any of the following is absent:

- one of fifty exact target checkpoints;
- one of sixty DPM sample sets, counting the frozen 300k baseline plus five continuations;
- one of four DDPM controls;
- PCA or SSCD metric tables for any checkpoint;
- physical summary, selected-bin, patch-boundary, or curve outputs;
- scheduler-terminal metadata;
- ten unique dataset tags per checkpoint.

- [ ] **Step 2: Run the audit tests and confirm failure**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
```

Expected: missing audit module or missing required checks.

- [ ] **Step 3: Implement the audit CLI and report**

Write a machine-readable report to:

```text
local/nf_generalize_fig2_dit_l16_continue500k_v2/final_audit.json
```

The report must include counts, missing paths, duplicate hashes, provenance mismatches, and a top-level `status` of `PASS` only when all required artifacts are valid.

- [ ] **Step 4: Wire PCA and SSCD jobs into the submitter**

For 300k, reuse the fresh-v2 tables if and only if their manifest and sample provenance point to the same source sweep; otherwise submit fresh 300k PCA and SSCD jobs rather than accepting or relabeling another table. For 340k through 500k, submit the existing PCA and SSCD Slurm scripts with explicit `MANIFEST_PATH`, `SAMPLE_LABEL`, and unique `OUT_PREFIX` values. Make the final audit depend on all PCA, SSCD, physical, DPM, and DDPM jobs.

- [ ] **Step 5: Add the audit Slurm wrapper**

The wrapper runs the CLI and exits nonzero unless `final_audit.json` reports `PASS`.

- [ ] **Step 6: Run tests and shell syntax checks**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continue500k_v2.py -q
python scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py --help
bash -n scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch
bash -n scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh
```

Expected: all tests pass and shell syntax checks are silent.

- [ ] **Step 7: Commit Task 6**

```bash
git add scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh tests/test_nf_fig2_dit_l16_continue500k_v2.py
git commit -m "Audit full DiT-L16 continuation results"
```

---

### Task 7: Integrate the continuation into the existing notebook

**Files:**
- Create: `scripts/update_dit_continue500k_v2_notebook.py`
- Modify: `notebooks/nf_generalize_fig2_dit_results.ipynb`
- Modify: `tests/test_dit_results_notebook_presentation.py`

- [ ] **Step 1: Add failing notebook-updater tests**

Use a temporary notebook fixture and the real notebook. Require the updater to:

- insert one contiguous block whose cells carry metadata tag `dit_l16_continue500k_v2`;
- replace that block, rather than duplicate it, on a second run;
- preserve every untagged cell byte-for-byte at the JSON-object level;
- keep the notebook filename unchanged;
- compile every code cell;
- contain no claims that continued training must produce the expected right-shifted scaling curve.

- [ ] **Step 2: Run the notebook tests and confirm failure**

Run:

```bash
python -m pytest tests/test_dit_results_notebook_presentation.py -q
```

Expected: missing updater or missing tagged section.

- [ ] **Step 3: Implement the idempotent notebook updater**

Use `nbformat`. Locate any existing tagged block, remove it, and insert the new section after the fresh-300k analysis. The section must load result files at execution time and show an explicit availability audit before plotting.

- [ ] **Step 4: Add concise collaborator-facing explanations and plots**

The new section must include:

1. **Provenance and availability table:** dataset tag, dataset size, target updates, checkpoint, sampler, steps, terminal audit, PCA table, SSCD table, and physics table.
2. **Loss trajectories:** all ten L16 datasets from 300k to 500k, aligned by optimizer updates, plus focused panels for `2^8` and `2^11`.
3. **Novelty transition:** PCA and SSCD q95 curves for L8 at 200k, L12 at 200k, and L16 at 300k, 340k, 380k, 420k, 460k, and 500k. The unequal budgets must be visible in title and legend.
4. **Physical sweep:** standalone one-point and power-spectrum figures covering all ten datasets with fixed axes and exact training-subset references.
5. **Selected-bin variance:** mean ratio and 95% interval at k bins 20, 40, and 60 versus optimizer updates for every dataset.
6. **Sampler control:** DPM-50 versus DDPM-500 for `2^8` and `2^11` at 300k and 500k.
7. **Patch audit:** patch-boundary ratio versus dataset size for L8, L12, and every L16 checkpoint, with a horizontal real-reference band.
8. **Nearest-training examples:** generated, nearest exact training slice, and absolute difference for `2^8` and `2^11` at 300k and 500k.
9. **Interpretation:** distinguish undertraining, sampler error, patch artifact, novelty, and physical fidelity; state that the scaling law is a hypothesis tested by the curves, not an enforced result.

Use separate figures where five- or ten-panel layouts would make labels unreadable. Put legends outside data regions and use consistent architecture colors and checkpoint line styles.

- [ ] **Step 5: Apply the updater to the existing dirty notebook**

Run:

```bash
python scripts/update_dit_continue500k_v2_notebook.py --notebook notebooks/nf_generalize_fig2_dit_results.ipynb
```

Expected: one tagged section is inserted; existing user cells and outputs remain.

- [ ] **Step 6: Verify idempotence and syntax**

Run the updater a second time, then:

```bash
python -m pytest tests/test_dit_results_notebook_presentation.py -q
python - <<'PY'
import ast, json
from pathlib import Path

path = Path('notebooks/nf_generalize_fig2_dit_results.ipynb')
notebook = json.loads(path.read_text())
for index, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code':
        ast.parse(''.join(cell.get('source', [])), filename=f'{path}:cell-{index}')
print('all notebook code cells compile')
PY
```

Expected: tests pass and the compile script prints `all notebook code cells compile`.

- [ ] **Step 7: Commit Task 7 without staging unrelated notebook files**

```bash
git add scripts/update_dit_continue500k_v2_notebook.py notebooks/nf_generalize_fig2_dit_results.ipynb tests/test_dit_results_notebook_presentation.py
git commit -m "Add DiT-L16 continuation diagnostics notebook"
```

---

### Task 8: Document, regress, and hand off the Great Lakes run

**Files:**
- Modify: `scripts/slurm/README.md`
- Modify only if required by test imports: `requirements.txt`

- [ ] **Step 1: Document the exact operator workflow**

Add:

```bash
cd /home/jiamingp/diffusion_models_repo
bash scripts/gl_safe_pull.sh main
bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh
```

Document:

- source sweep and checkpoint root;
- five targets and expected labels;
- 48-hour tasks and `%2` limit;
- exact restart command using `START_STAGE` and `REUSE_EXISTING_MANIFEST=1`;
- progress, log, and final-audit commands;
- result table and notebook paths;
- the four DDPM controls;
- the rule that no scientific conclusion is valid until the final audit passes.

- [ ] **Step 2: Run the focused regression suite**

Run:

```bash
python -m pytest \
  tests/test_nf_fig2_dit_l16_continue500k_v2.py \
  tests/test_dit_l16_diagnostics.py \
  tests/test_nf_fig2_continuation_guards.py \
  tests/test_dit_results_notebook_presentation.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run the broader relevant regression suite**

Run:

```bash
python -m pytest \
  tests/test_dit_checkpoint_resume.py \
  tests/test_nf_fig2_dit_l16_continuation.py \
  tests/test_real_reference_loading.py \
  tests/test_feature_distribution_distance.py \
  tests/test_nearest_training_matches.py -q
```

Expected: all existing resume, reference, and metric tests pass.

- [ ] **Step 4: Run static and shell checks**

Run:

```bash
python -m compileall -q scripts simdiff_eval tests
for file in \
  scripts/slurm/precheck_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch \
  scripts/slurm/train_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch \
  scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch \
  scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_ddpm_controls.sbatch \
  scripts/slurm/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.sbatch \
  scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch \
  scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh; do
  bash -n "$file"
done
```

Expected: no output and zero exit status.

- [ ] **Step 5: Review the diff against the approved design**

Confirm explicitly:

- all ten datasets are present at every target;
- source sweep is fresh300k v2 only;
- full-state resume is required;
- patch/preprocessing parity is checked before GPU training;
- DPM and DDPM outputs cannot collide;
- terminal scheduler metadata is saved;
- exact training-subset references are used;
- selected-bin variance and patch-boundary metrics are included;
- notebook update is idempotent and preserves existing work;
- no code filters, replaces, or relabels results to manufacture a scaling relation.

- [ ] **Step 6: Commit documentation and any final test-only corrections**

```bash
git add scripts/slurm/README.md
git commit -m "Document DiT-L16 continuation workflow"
```

- [ ] **Step 7: Great Lakes handoff verification**

After pulling on Great Lakes, run the preparation and precheck only first:

```bash
cd /home/jiamingp/diffusion_models_repo
python scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py --project-dir .
python scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py --project-dir . --use-existing-manifest --check-only
```

Inspect the generated manifest and confirm fifty continuation rows, ten datasets per target, and source checkpoints under the fresh300k-v2 root. Only then submit the Slurm chain.
