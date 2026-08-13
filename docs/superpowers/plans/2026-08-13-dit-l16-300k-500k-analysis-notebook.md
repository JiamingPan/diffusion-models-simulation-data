# DiT-L16 300k-500k Analysis Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained audited notebook covering the complete DiT-L16 300k-500k continuation analysis.

**Architecture:** A deterministic Python builder owns the notebook JSON and embeds self-contained analysis cells. Structural tests inspect and compile every code cell, enforce the audited data contract, and prevent legacy L16 200k material from returning.

**Tech Stack:** Python, Jupyter notebook JSON, pandas, NumPy, Matplotlib, pytest, existing `simdiff_eval` and `scripts.dit_300k_scaling_analysis` helpers.

## Global Constraints

- Core L16 checkpoints are exactly 300k, 340k, 380k, 420k, 460k, and 500k.
- Dataset sizes are exactly `2^6` through `2^15`.
- The final artifact audit must pass before any result is plotted.
- Historical UNet and DiT-L8/L12 curves may appear only in the explicitly labeled context section.
- No old L16 200k result may appear.
- Repository notebook remains unexecuted because result arrays are only on Great Lakes.

---

### Task 1: Define the notebook contract

**Files:**
- Create: `tests/test_dit_l16_300k_500k_analysis_notebook.py`

**Interfaces:**
- Consumes: planned builder path and notebook path.
- Produces: executable contract for required sections, data guards, and compiled code cells.

- [x] Write tests requiring the builder and notebook.
- [x] Require all fifteen approved analysis sections and six checkpoints.
- [x] Require all ten dataset tags and reject legacy L16 200k identifiers.
- [x] Require every code cell to compile and every output to be empty.
- [x] Run the test and confirm it fails because the builder is absent.

### Task 2: Build the deterministic notebook

**Files:**
- Create: `scripts/build_dit_l16_300k_500k_analysis_notebook.py`
- Create: `notebooks/nf_generalize_fig2_dit_l16_300k_500k_analysis.ipynb`

**Interfaces:**
- Consumes: audited continuation manifests and existing validated helper modules.
- Produces: `build_notebook() -> dict` and `main() -> None`.

- [x] Add deterministic cell IDs and notebook metadata.
- [x] Add standalone setup, audit, and checkpoint-specific table validation.
- [x] Add optimization, generated-map, novelty, transition, and context sections.
- [x] Add one-point, power-spectrum, selected-bin, sampler, patch, nearest-neighbor, and joint verification sections.
- [x] Generate the notebook and run the contract tests.

### Task 3: Validate the notebook and documentation

**Files:**
- Modify: `notebooks/README.md`
- Test: `tests/test_dit_l16_300k_500k_analysis_notebook.py`

**Interfaces:**
- Consumes: generated notebook.
- Produces: documented Great Lakes execution command and verified repository artifact.

- [x] Document the notebook scope and Great Lakes execution command.
- [x] Regenerate the notebook twice and confirm byte-for-byte determinism.
- [x] Run focused tests and relevant continuation tests.
- [x] Inspect notebook section order, metadata, and absence of stored outputs.
- [x] Commit the completed local branch without pushing.
