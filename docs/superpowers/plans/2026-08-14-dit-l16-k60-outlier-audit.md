# DiT-L16 k=60 Outlier Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative, fully auditable k=60 outlier sensitivity analysis to the DiT-L16 300k-500k notebook.

**Architecture:** Put reusable robust-selection and summary logic in `scripts/dit_300k_scaling_analysis.py`. The notebook builder loads each existing 512-sample archive, computes per-sample spectra, uses the exact model-specific real reference already loaded by the notebook, and renders both unfiltered and filtered diagnostics without modifying source artifacts.

**Tech Stack:** Python, NumPy, pandas, Matplotlib, pytest, nbformat.

## Global Constraints

- Use a two-sided 4.5-MAD rule in log10 k=60 ratio space.
- Do not flag samples when MAD is zero or non-finite.
- Preserve unfiltered results as the primary analysis.
- Report excluded sample identities and retained counts.
- Do not submit training, sampling, or analysis jobs.

---

### Task 1: Robust outlier selection

**Files:**
- Modify: `scripts/dit_300k_scaling_analysis.py`
- Test: `tests/test_dit_300k_scaling_analysis.py`

**Interfaces:**
- Produces: `robust_log_ratio_outliers(values, threshold=4.5) -> dict[str, np.ndarray | float]`
- Produces: `summarize_filtered_power_ratios(pk_ratio, outlier_mask, bin_indices) -> list[dict[str, float | int]]`

- [ ] Write tests proving an extreme high ratio is flagged, ordinary values are retained, and zero MAD flags nothing.
- [ ] Run the focused tests and confirm they fail because the functions do not exist.
- [ ] Implement input validation, log transform, median/MAD scoring, and filtered summaries.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Notebook audit and plots

**Files:**
- Modify: `scripts/build_dit_l16_300k_500k_analysis_notebook.py`
- Regenerate: `notebooks/nf_generalize_fig2_dit_l16_300k_500k_analysis.ipynb`
- Test: `tests/test_dit_l16_300k_500k_analysis_notebook.py`

**Interfaces:**
- Consumes: the robust outlier helpers from Task 1.
- Produces: in-notebook `outlier_samples`, `outlier_groups`, and `filtered_selected_bins` tables plus five saved figures.

- [ ] Write structural tests requiring the rule, audit tables, flagged-map gallery, retained counts, filtered selected-bin comparison, and filtered 500k spectra.
- [ ] Run the notebook tests and confirm they fail because those sections are absent.
- [ ] Add notebook markdown that distinguishes primary and sensitivity analyses.
- [ ] Add code that loads every existing DPM50 archive, computes per-sample spectra, applies the k=60 rule, and builds the audit tables.
- [ ] Add the per-sample distribution, flagged-map gallery, selected-bin comparison, and 500k full-spectrum figures.
- [ ] Regenerate the notebook and run the structural tests.

### Task 3: Verification

**Files:**
- Verify only; no new files required.

**Interfaces:**
- Consumes: all changes from Tasks 1 and 2.
- Produces: test evidence and a local commit.

- [ ] Compile all notebook code cells.
- [ ] Run the targeted helper and notebook tests.
- [ ] Run the broader DiT continuation analysis test suite.
- [ ] Inspect the git diff for unrelated changes.
- [ ] Commit the verified implementation locally.
- [ ] Preview the exact push command and wait for `APPROVE PUSH` before changing the remote branch.
