# DiT-L16 Seed-Restart Safety Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the seed-456 DiT-L16 continuation consume an immutable, audited cosmodiff pin and make every precheck report fail closed through an atomic terminal-report lifecycle.

**Architecture:** Build the patched cosmodiff source tree once outside Slurm, publish it atomically with a manifest, and verify that manifest at job start before inspecting continuation data. Add a small shared terminal-report module plus a CLI finalizer; Python producers write only an `INCOMPLETE` payload, and Slurm wrappers alone finalize `PASS` or `FAILED` with job identity and exit code.

**Tech Stack:** Python 3.10, pathlib, hashlib, subprocess, JSON, Bash/Slurm, pytest, AST-based source-policy tests.

---

### Task 1: Specify the terminal-report state machine with failing tests

**Files:**
- Create: `tests/test_terminal_reports.py`
- Create: `simdiff_eval/terminal_reports.py`
- Create: `scripts/finalize_terminal_report.py`

**Step 1: Write failing lifecycle tests**

Add tests that call the intended API:

```python
report = terminal_reports.start_report(
    path,
    payload={"rows": [{"dataset_tag": "d2p08"}]},
    producer_job_id="12345",
)
assert report["status"] == "INCOMPLETE"
assert report["producer_exit_code"] is None
assert report["finalized_at_utc"] is None

passed = terminal_reports.finalize_report(
    path,
    status="PASS",
    producer_job_id="12345",
    producer_exit_code=0,
)
terminal_reports.require_passed_report(path, expected_producer_job_id="12345")
```

Cover these failures explicitly: PASS with nonzero exit code, FAILED with zero exit code, mismatched job ID, legacy/missing schema, second terminal finalization, malformed timestamps, and a temporary file that must never be accepted as the report. Monkeypatch `os.replace` to prove readers see either the old complete JSON or the new complete JSON, never a partial write.

**Step 2: Run the new test and confirm RED**

Run: `pytest -q tests/test_terminal_reports.py`

Expected: import failure for `simdiff_eval.terminal_reports`.

**Step 3: Implement the shared atomic module**

Create `simdiff_eval/terminal_reports.py` with:

```python
REPORT_SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"PASS", "FAILED", "STALE"})

def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None: ...
def start_report(path: Path, *, payload: Mapping[str, Any], producer_job_id: str | None) -> dict[str, Any]: ...
def finalize_report(path: Path, *, status: str, producer_job_id: str | None, producer_exit_code: int) -> dict[str, Any]: ...
def mark_stale(path: Path, *, reason: str) -> dict[str, Any]: ...
def require_passed_report(path: Path, *, expected_producer_job_id: str | None) -> dict[str, Any]: ...
```

Use an adjacent uniquely named temporary file, `flush()` and `os.fsync()`, then `os.replace()`. Preserve the producer payload while changing only lifecycle fields. `PASS` requires exit code 0; `FAILED` requires nonzero; `STALE` records a reason. A reader requires schema 1, status PASS, exit 0, non-null start/final timestamps, and exact producer job identity.

**Step 4: Implement the CLI finalizer**

Create `scripts/finalize_terminal_report.py` with subcommands:

```text
finalize PATH --status PASS|FAILED --job-id JOB --exit-code N
require-pass PATH --expected-job-id JOB
mark-stale PATH --reason TEXT
```

The CLI imports only the shared module, emits the validated JSON on success, and returns nonzero on invalid transitions.

**Step 5: Run tests and commit**

Run: `pytest -q tests/test_terminal_reports.py`

Expected: PASS.

Commit:

```bash
git add simdiff_eval/terminal_reports.py scripts/finalize_terminal_report.py tests/test_terminal_reports.py
git commit -m "feat: add atomic terminal report lifecycle"
```

### Task 2: Build an immutable patched cosmodiff pin

**Files:**
- Create: `scripts/build_cosmodiff_seed_restart_pin.py`
- Replace: `scripts/verify_cosmodiff_seed_restart_runtime.py`
- Modify: `tests/test_nf_fig2_dit_l16_seed_restart500k.py`

**Step 1: Write failing pin-builder tests**

Build a tiny local Git repository fixture with `cosmodiff/__init__.py`, `optim.py`, `utils.py`, `augment.py`, and `transform.py`. Monkeypatch or provide four small patch executables in the declared order. Assert the builder:

- checks out the exact requested base revision into staging;
- applies exactly `patch_cosmodiff_package_metadata.py`, `patch_cosmodiff_constant_label.py`, `patch_cosmodiff_dit_class_labels.py`, then `patch_cosmodiff_checkpoint_state.py`;
- records each patch script SHA256, `applied` or `already_supported`, and target before/after hashes;
- removes `.codex_*.bak` files;
- records exact imported module paths and `cosmodiff.__version__` using the requested interpreter;
- records an exact final file inventory;
- publishes with one atomic rename only after verification;
- refuses an existing destination, wrong base revision, a failed patch/import, extra untracked file, modified target, or changed patch-script hash.

**Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_nf_fig2_dit_l16_seed_restart500k.py -k 'pin or immutable'`

Expected: missing builder/verifier APIs.

**Step 3: Implement the builder**

Create `scripts/build_cosmodiff_seed_restart_pin.py` with reusable functions and CLI:

```text
build_cosmodiff_seed_restart_pin.py \
  --source-repo SOURCE \
  --base-revision REV \
  --destination DEST \
  --python-bin PYTHON \
  --patch-script SCRIPT   # exactly four, order validated
```

Use `git worktree add --detach` or a local clone into `DEST.tmp-<pid>`. Hash target files before and after every patch. Determine `applied` from target-hash changes, otherwise `already_supported`. Delete only the known backup sidecars created in staging. Import all five required modules in a subprocess whose `PYTHONPATH` starts with staging, verify every resolved module path lies inside staging, write `seed_restart_pin_manifest.json`, verify it, and atomically rename staging to DEST.

**Step 4: Implement strict pin verification**

Replace fragment-only verification with reusable `verify_pin(pin_root, manifest_path, expected_base_revision, python_bin)`. Validate the base revision, ordered patch names and script hashes, exact inventory, every final SHA256, no missing or extra file, required imports and their recorded paths/version, and the external interpreter identity. Keep the existing source-contract fragment checks as an additional validation after the manifest checks.

**Step 5: Run tests and commit**

Run: `pytest -q tests/test_nf_fig2_dit_l16_seed_restart500k.py -k 'pin or immutable or source_contract'`

Expected: PASS.

Commit:

```bash
git add scripts/build_cosmodiff_seed_restart_pin.py scripts/verify_cosmodiff_seed_restart_runtime.py tests/test_nf_fig2_dit_l16_seed_restart500k.py
git commit -m "feat: build audited cosmodiff restart pin"
```

### Task 3: Make seed-restart prechecks lifecycle-safe

**Files:**
- Modify: `scripts/check_nf_generalize_fig2_dit_l16_seed_restart500k.py`
- Modify: `scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch`
- Modify: `scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch`
- Modify: `scripts/slurm/submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh`
- Modify: `tests/test_nf_fig2_dit_l16_seed_restart500k.py`

**Step 1: Write failing producer/consumer tests**

Assert the checker writes an INCOMPLETE payload without a literal terminal PASS. Assert the precheck shell installs an EXIT trap before work, finalizes PASS only after the pin import verification, manifest validation, checkpoint reconstruction, and resume dry run all succeed, and finalizes FAILED on a normal error. Assert training calls `require-pass` with `${SLURM_JOB_DEPENDENCY#afterok:}` and refuses a report from any other producer job. Assert submission exports `COSMODIFF_PIN_ROOT` and `COSMODIFF_PIN_MANIFEST` instead of a mutable checkout plus ad-hoc dist-info stub.

**Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_nf_fig2_dit_l16_seed_restart500k.py -k 'report or precheck or submit or train'`

Expected: current scripts still embed PASS and mutable-checkout logic.

**Step 3: Change the Python checker to produce INCOMPLETE only**

After all row validation succeeds, call `start_report()` with row diagnostics and the job ID. Do not print or store PASS. Preserve `--report` fail-closed non-overwrite behavior.

**Step 4: Change the precheck shell to own finalization**

At startup, set `REPORT_PATH`, `PRODUCER_JOB_ID=${SLURM_JOB_ID}`, and an EXIT trap that calls:

```bash
finalize_terminal_report.py finalize "$REPORT_PATH" \
  --status FAILED --job-id "$PRODUCER_JOB_ID" --exit-code "$rc"
```

only if a report exists and is still INCOMPLETE. After every required command and dry run succeeds, call the same CLI with `--status PASS --exit-code 0`, then disarm the trap. Verify the immutable pin before any manifest/checkpoint/report operation.

**Step 5: Change training and submission consumers**

Training requires the terminal report using the exact expected precheck job ID exported by submission. Both precheck and training use only the verified pin root on `PYTHONPATH`. Submission validates the pin locally before calling `sbatch` and exports the exact pin manifest/hash to all stages.

**Step 6: Run tests and commit**

Run: `pytest -q tests/test_nf_fig2_dit_l16_seed_restart500k.py tests/test_terminal_reports.py`

Expected: PASS.

Commit:

```bash
git add scripts/check_nf_generalize_fig2_dit_l16_seed_restart500k.py scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch scripts/slurm/submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh tests/test_nf_fig2_dit_l16_seed_restart500k.py
git commit -m "fix: make seed restart reports fail closed"
```

### Task 4: Migrate remaining terminal-report producers and enforce policy

**Files:**
- Modify: `scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py`
- Modify: `scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch`
- Modify: `tests/test_nf_fig2_dit_l16_continue500k_v2.py`
- Create: `tests/test_terminal_report_policy.py`

**Step 1: Write the AST policy test**

Parse Python ASTs and shell-embedded Python ASTs for the allowlisted terminal-report producers. Fail if a producer constructs `{"status": "PASS"}` or assigns terminal PASS without importing/calling `finalize_report` or the finalizer CLI. Explicitly exclude configuration manifests, metric payloads, and ordinary metadata writers from the producer allowlist.

**Step 2: Run policy test and confirm RED**

Run: `pytest -q tests/test_terminal_report_policy.py`

Expected: continuation audit producer is flagged.

**Step 3: Migrate the continuation audit**

Make the Python audit start INCOMPLETE, let its Slurm wrapper finalize PASS/FAILED, and make downstream consumers require the new schema. Do not change scientific metric JSON payloads.

**Step 4: Run focused tests and commit**

Run: `pytest -q tests/test_terminal_report_policy.py tests/test_nf_fig2_dit_l16_continue500k_v2.py`

Expected: PASS.

Commit:

```bash
git add scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch tests/test_nf_fig2_dit_l16_continue500k_v2.py tests/test_terminal_report_policy.py
git commit -m "test: enforce terminal report finalization policy"
```

### Task 5: Verify the DiT workflow and prepare—but do not execute—the Great Lakes payload

**Files:**
- Modify only if verification exposes a defect: files changed in Tasks 1–4

**Step 1: Run focused Python and shell checks**

Run:

```bash
python -m compileall -q simdiff_eval scripts
bash -n scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch
bash -n scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch
bash -n scripts/slurm/submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh
pytest -q tests/test_terminal_reports.py tests/test_terminal_report_policy.py tests/test_nf_fig2_dit_l16_seed_restart500k.py tests/test_dit_checkpoint_resume.py
```

Expected: all PASS.

**Step 2: Run the complete suite**

Run: `pytest -q`

Expected: all tests PASS.

**Step 3: Review repository state**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -8
```

Expected: no whitespace errors; only intended commits.

**Step 4: Prepare external-action previews**

Draft, but do not run, one exact push preview and one exact Great Lakes payload that: fetches the exact delivered commit, creates/verifies the pin, marks only `stage1_58470485.json` STALE with its reason, proves no continuation train job is currently active for the same target tree, and submits stages 1–5 once. State GPU count, wall-time ceilings, quota impact, output paths, and that approval must be `APPROVE PUSH` then separately `APPROVE RUN`.
