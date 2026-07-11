# DiT-L16 Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the five small-data DiT-L16 runs by 100k optimizer updates in resumable 25k stages and evaluate exact stage checkpoints against their complete configured training references.

**Architecture:** A Python config generator reads existing checkpoint epochs and writes four absolute-epoch continuation configs per run. Slurm arrays execute one stage at a time with `afterok` dependencies, saving safety checkpoints every 5k updates. Exact stage checkpoints are sampled with unique labels; the notebook loads the complete configured training reference and asserts its slice count.

**Tech Stack:** Python 3.10, PyYAML, pytest, Slurm, PyTorch/diffusers through `cosmo_diffusion`, Jupyter JSON.

## Global Constraints

- Continue only `dit_l16` runs `d2p06` through `d2p10`.
- Add exactly four nominal 25k-update stages, 100k total.
- Keep the existing data, model, optimizer, LR scheduler, EMA, sampler, and seed.
- Use eight-hour stage walltimes and safety checkpoints about every 5k updates.
- Sampling must name an exact checkpoint path, never an implicitly latest directory.
- The controlled UNet reference must contain all configured training slices.

---

### Task 1: Continuation arithmetic and configs

**Files:**
- Create: `scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py`
- Create: `tests/test_nf_fig2_dit_l16_continuation.py`

**Interfaces:**
- Consumes: `prepare_nf_generalize_fig2_dit_configs.iter_runs()` and existing checkpoint directories.
- Produces: `continue_rows(args)`, stage YAML files, and `local/nf_generalize_fig2_dit_l16_continue/manifest.json`.

- [ ] Write failing tests for five runs, four stages, cumulative update arithmetic, absolute final epochs, and DiT-L16 config invariants.
- [ ] Run `pytest -q tests/test_nf_fig2_dit_l16_continuation.py` and verify failure because the generator is absent.
- [ ] Implement checkpoint discovery, update-to-epoch conversion, stage rows, YAML output, and `--check-only`/`--print-table` modes.
- [ ] Re-run the focused tests and verify they pass.

### Task 2: Resumable Slurm training and exact sampling

**Files:**
- Create: `scripts/slurm/train_nf_generalize_fig2_dit_l16_continue_array.sbatch`
- Create: `scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue_array.sbatch`
- Create: `scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue.sh`
- Modify: `tests/test_nf_fig2_dit_l16_continuation.py`

**Interfaces:**
- Training consumes `CONTINUE_STAGE=1..4` and array task `0..4`.
- Sampling consumes the same stage and requires `expected_checkpoint` from the manifest.
- Submission emits four dependent training arrays and four dependent sampling arrays.

- [ ] Add failing static tests for eight-hour walltime, 5k safety interval, stage selection, exact checkpoint sampling, and `afterok` dependencies.
- [ ] Implement the two array wrappers using the established DiT environment/runtime checks.
- [ ] Implement the submission script with `%2` concurrency and stage dependencies.
- [ ] Run shell syntax checks and the focused pytest file.

### Task 3: Full-reference fidelity correction

**Files:**
- Modify: `notebooks/nf_generalize_fig2_partial_quickcheck.ipynb`
- Modify: `tests/test_nf_fig2_continuation_guards.py`

**Interfaces:**
- Consumes the original run config through `load_real_from_config(config_path, max_raw_samples=None)`.
- Produces a comparison table containing `real_reference_kind`, `real_config_path`, and `n_real`, with `n_real == dataset_size`.

- [ ] Add a failing notebook-source test requiring a complete training reference and count assertion.
- [ ] Replace the capped `loaded` bundle in the controlled comparison with a fresh full-config load.
- [ ] Add provenance fields and fail if the real slice count differs from the manifest dataset size.
- [ ] Validate notebook JSON and compile every code cell.

### Task 4: Verification and delivery

**Files:**
- Modify: `scripts/slurm/README.md`

- [ ] Document precheck, config generation, staged submission, monitoring, and analysis commands.
- [ ] Run focused and existing continuation tests.
- [ ] Run `bash -n` on all new Slurm/shell scripts.
- [ ] Run generator smoke tests in a temporary checkpoint tree.
- [ ] Commit only the continuation workflow, notebook correction, tests, and documentation; push `main`.
