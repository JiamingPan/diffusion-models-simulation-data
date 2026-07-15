# DiT-L16 Resume Target Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every DiT-L16 continuation stage stop at its exact 225k, 250k, 275k, or 300k optimizer-update checkpoint under both absolute-epoch and additional-epoch `cosmodiff` resume semantics.

**Architecture:** Generate a versioned manifest that separates immutable original checkpoints from clean continuation directories, then seed each continuation directory with the exact 200k checkpoint. At runtime, inspect the installed trainer's epoch-loop semantics, convert the manifest's absolute target into the correct argument, and reject contaminated or overshot state before launching GPU training.

**Tech Stack:** Python 3, PyYAML, pytest, Bash, Slurm, diffusers/cosmodiff checkpoint directories.

## Global Constraints

- Preserve all existing original and overshot checkpoints; never delete or rewrite them.
- Use one isolated continuation checkpoint directory per DiT-L16 run.
- Run four sequential stages of nominally 25,000 additional optimizer updates.
- Sample and analyze only the exact manifest checkpoint for each stage.
- Reject old unversioned continuation manifests.
- Follow red-green-refactor for every behavior change.

---

### Task 1: Versioned Isolated Continuation Manifest

**Files:**
- Modify: `tests/test_nf_fig2_dit_l16_continuation.py`
- Modify: `scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py`

**Interfaces:**
- Produces: `continue_rows(args) -> list[dict[str, Any]]` with `manifest_version`, `base_checkpoint`, `previous_expected_checkpoint`, `expected_checkpoint`, and isolated `checkpoint_dir` fields.
- Produces: `seed_continuation_directories(rows) -> None`, which creates validated symlinks to exact 200k checkpoints.

- [x] **Step 1: Write failing manifest and seeding tests**

Add tests that create original checkpoints including an overshot checkpoint, then assert the manifest still selects `base_row["epochs"] - 1`, writes to a separate continuation root, computes stage targets cumulatively, and rejects a conflicting seed.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continuation.py -q
```

Expected: failures for missing `--continuation-checkpoint-root`, versioned fields, and seed helper.

- [x] **Step 3: Implement isolated manifest arithmetic**

Use absolute checkpoint identities:

```python
base_epoch = int(base_row["epochs"]) - 1
previous_epoch = base_epoch + math.ceil((stage - 1) * stage_updates / steps_per_epoch)
target_epoch = base_epoch + math.ceil(stage * stage_updates / steps_per_epoch)
stage_additional_epochs = target_epoch - previous_epoch
```

Write original checkpoints and continuation checkpoints to distinct roots, set `manifest_version = 2`, and seed only the exact base checkpoint into the clean directory.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2 and require zero failures.

- [x] **Step 5: Commit Task 1**

```bash
git add tests/test_nf_fig2_dit_l16_continuation.py scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py
git commit -m "Fix DiT continuation manifest targets"
```

### Task 2: Runtime Epoch-Semantics Adapter

**Files:**
- Modify: `tests/test_dit_checkpoint_resume.py`
- Modify: `scripts/run_cosmodiff_train_with_dit_resume.py`

**Interfaces:**
- Produces: `detect_epoch_semantics(train_fn) -> str`, returning `"absolute"` or `"additional"`.
- Produces: `epoch_argument(start_epoch: int, target_epoch: int, semantics: str) -> int`.
- CLI consumes: `--checkpoint-dir` and `--target-checkpoint`.

- [x] **Step 1: Write failing regression tests**

Cover the observed case exactly:

```python
assert epoch_argument(12792, 14062, "additional") == 1271
assert epoch_argument(12792, 14062, "absolute") == 14063
```

Also test trainer-loop detection, exact-target no-op, and rejection when the latest checkpoint is beyond the target.

- [x] **Step 2: Run the resume tests and verify RED**

```bash
python -m pytest tests/test_dit_checkpoint_resume.py -q
```

Expected: failures because the adapter and new CLI contract do not exist.

- [x] **Step 3: Implement the adapter**

Patch `cosmodiff.optim.train` in process. The adapter receives the external script's computed `start_epoch`, detects its loop semantics from source, and replaces `num_epochs` with either `target_epoch + 1` for absolute semantics or `target_epoch + 1 - start_epoch` for additional semantics. Before running, require `latest_epoch <= target_epoch`; return without training only when they are equal.

- [x] **Step 4: Run the resume tests and verify GREEN**

Run the command from Step 2 and require zero failures.

- [x] **Step 5: Commit Task 2**

```bash
git add tests/test_dit_checkpoint_resume.py scripts/run_cosmodiff_train_with_dit_resume.py
git commit -m "Adapt DiT resume epochs to trainer semantics"
```

### Task 3: Slurm Contract And Migration Guard

**Files:**
- Modify: `tests/test_nf_fig2_dit_l16_continuation.py`
- Modify: `scripts/slurm/train_nf_generalize_fig2_dit_l16_continue_array.sbatch`
- Modify: `scripts/slurm/precheck_nf_generalize_fig2_dit_l16_resume.sbatch`
- Modify: `scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue.sh`

**Interfaces:**
- Train wrapper passes exact `checkpoint_dir` and `expected_checkpoint` to the runtime adapter.
- Precheck loads the manifest's exact `previous_expected_checkpoint`.
- Submit script creates/seeds a version-2 manifest and refuses the failed version-1 manifest.

- [x] **Step 1: Write failing static workflow tests**

Assert that the submission script requests seeding, the precheck reads `previous_expected_checkpoint`, and the train script passes both new runtime arguments.

- [x] **Step 2: Run workflow tests and verify RED**

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continuation.py -q
```

Expected: static assertions fail against the old Slurm scripts.

- [x] **Step 3: Update the Slurm workflow**

Generate and seed the new manifest before submission, validate version 2 during reuse, precheck the exact previous checkpoint, and pass:

```bash
--checkpoint-dir "${CHECKPOINT_DIR}" \
--target-checkpoint "${EXPECTED_CHECKPOINT}"
```

to the class-safe runtime wrapper.

- [x] **Step 4: Verify shell and focused tests**

```bash
bash -n scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue.sh
bash -n scripts/slurm/train_nf_generalize_fig2_dit_l16_continue_array.sbatch
bash -n scripts/slurm/precheck_nf_generalize_fig2_dit_l16_resume.sbatch
python -m pytest tests/test_nf_fig2_dit_l16_continuation.py tests/test_dit_checkpoint_resume.py -q
```

Expected: shell parsing succeeds and all focused tests pass.

- [x] **Step 5: Commit Task 3**

```bash
git add tests/test_nf_fig2_dit_l16_continuation.py scripts/slurm/train_nf_generalize_fig2_dit_l16_continue_array.sbatch scripts/slurm/precheck_nf_generalize_fig2_dit_l16_resume.sbatch scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue.sh
git commit -m "Enforce exact DiT stage checkpoints"
```

### Task 4: End-To-End Verification And Great Lakes Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-15-dit-l16-resume-target-fix.md`

**Interfaces:**
- Produces: verified local test evidence and a fresh Great Lakes submission command that does not reuse the failed manifest.

- [x] **Step 1: Run the full continuation test suite**

```bash
python -m pytest tests/test_nf_fig2_dit_l16_continuation.py tests/test_dit_checkpoint_resume.py -q
```

- [x] **Step 2: Generate a temporary manifest and inspect target arithmetic**

Create temporary original checkpoint directories, generate a new manifest with a temporary continuation root, and confirm the d2p07 stage-1 row has base epoch 12499, target epoch 14062, and 1563 stage epochs while the runtime regression computes 1271 remaining epochs from safety epoch 12791.

- [x] **Step 3: Review the diff and scan for stale semantics**

```bash
git diff --check
git grep -n "latest_checkpoint_epoch_at_prepare\|resume_start_epoch" -- scripts tests
```

Expected: no whitespace errors and no production dependence on the contaminated latest original checkpoint.

- [x] **Step 4: Record verification and commit**

Mark completed checklist items in this plan and commit the verified state.

## Verification Record

- Full local suite: `32 passed in 8.28s`.
- Shell syntax: submission, training, and resume-precheck scripts parsed successfully.
- Old additional-epoch trainer simulation: resume start `12792`, target `14062`, runtime argument `1271`, exact target created, no overshoot.
- Five-run manifest simulation: version 2, isolated checkpoint roots, exact 200k seeds, and all four stage targets validated.
