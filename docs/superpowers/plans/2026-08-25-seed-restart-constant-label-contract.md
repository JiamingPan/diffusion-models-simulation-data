# Seed-Restart Constant-Label Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DiT seed-restart label adapter idempotent while preserving its fail-closed protection and recording exact pin provenance.

**Architecture:** Keep the compatibility behavior in the existing in-process adapter. Classify the returned dataset into injected, already-correct, or conflicting paths, emit one structured line, and leave correct existing labels untouched. Record base-versus-effective support in schema-3 pin manifests and verify the record against the declared patch result.

**Tech Stack:** Python 3.10, PyTorch, pytest/unittest, JSON pin manifests, Bash/Slurm launchers.

**Spec:** `docs/superpowers/specs/2026-08-25-seed-restart-constant-label-contract.md`

## Global Constraints

- Do not modify the 300k source checkpoints.
- Do not touch C4 work or results.
- Do not push or submit Slurm jobs.
- Preserve the conditional-label guard.
- Finish with green tests and a clean `git diff --check`.

---

### Task 1: Three-way constant-label adapter

**Files:**
- Modify: `tests/test_dit_checkpoint_resume.py`
- Modify: `scripts/run_cosmodiff_train_with_dit_resume.py`

**Interfaces:**
- Consumes: `install_constant_label_adapter(utils_module)` and an ArrayDataset-like object with `arrays` and `labels`.
- Produces: one of `legacy_injected`, `existing_constant_noop`, or `conflict_refused`, printed with dtype, length, and unique values.

- [x] Add separate failing tests for missing labels, matching existing labels, and genuine multivalued labels.
- [x] Run the three tests and confirm the matching-label test fails with the production exception and the log assertions fail.
- [x] Implement the smallest three-way classifier and flushed provenance log.
- [x] Run the focused tests and retain the real-label exception.

### Task 2: Immutable-pin support provenance

**Files:**
- Modify: `tests/test_nf_fig2_dit_l16_seed_restart500k.py`
- Modify: `scripts/build_cosmodiff_seed_restart_pin.py`
- Modify: `scripts/verify_cosmodiff_seed_restart_runtime.py`

**Interfaces:**
- Consumes: the unmodified staged `cosmodiff/utils.py` and the ordered constant-label patch record.
- Produces: schema-3 `constant_label_support` with `native_in_base_revision`, `effective_in_published_pin`, and `provenance`.

- [x] Add failing builder/verifier assertions for a base revision without native support and for a native-support fixture.
- [x] Run the focused tests and confirm the manifest field is absent.
- [x] Detect support before patches, record its effective provenance after patches, and make the verifier fail closed on inconsistency.
- [x] Run builder/verifier tests.

### Task 3: Downstream compatibility audit

**Files:**
- Create: `docs/superpowers/reports/2026-08-25-seed-restart-first-step-audit.md`

**Interfaces:**
- Consumes: frozen Cosmodiff `58c77eb`, all four pin patches, wrapper hooks, preflight evidence, and the failed job logs supplied by the user.
- Produces: an ordered parse-to-first-optimizer-step audit listing every older-API assumption and its disposition.

- [x] Trace data parse, checkpoint discovery, exact-target binding, state restore, RNG reset, EMA restore, model forward, backward audit, and optimizer step.
- [x] Record the corrected finding that constant-label support is patch-provided rather than native in the base revision.
- [x] List remaining risks and whether each is already guarded, tested, or still requires a Great Lakes smoke run.

### Task 4: Verification

**Files:**
- Verify all modified Python, test, documentation, shell, and Slurm files.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: local evidence only; no external action.

- [x] Run focused tests.
- [x] Run the full test suite.
- [x] Run `python -m compileall` on changed Python files.
- [x] Run `bash -n` on changed shell and sbatch files, if any.
- [x] Run `git diff --check` and confirm C4 files are unchanged.
