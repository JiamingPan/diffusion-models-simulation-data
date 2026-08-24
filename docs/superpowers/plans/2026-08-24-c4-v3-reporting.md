# C4 v3 Reporting Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a fresh, non-overwriting C4 v3 analysis whose headline claims live only in frozen standardized 1024-D VGG/MLP input space, explicitly identifies no-op transforms, and reports exact and empirical mixing baselines.

**Architecture:** Extend the pure metric module with identity, power-deficit, balanced-split, and perfect-mixing helpers; keep the evaluator as the orchestration layer that retains source arrays, fits UMAP once for visualization, separates complete/headline/visual-only artifacts, and finalizes its manifest through the shared terminal-report lifecycle. The existing v2 output remains untouched.

**Tech Stack:** NumPy, pandas, scikit-learn, UMAP 0.5.5, Matplotlib, PyTorch/VGG16, JSON/CSV/NPZ, pytest, Bash/Slurm.

---

### Task 1: Add exact power-deficit and identity diagnostics

**Files:**
- Modify: `simdiff_eval/probe_c4_umap.py`
- Modify: `tests/test_probe_c4_umap.py`

**Step 1: Write failing helper tests**

Add tests for:

```python
assert measured_power_deficit_depth([1.02, 0.65, np.nan]) == pytest.approx(0.35)
assert measured_power_deficit_depth([1.02, 1.01]) == 0.0

identity = classify_transform_identity(
    original.astype(np.float32),
    transformed.astype(np.float32),
    metric={
        "centroid_distance_ci_low": 0.0,
        "centroid_distance_ci_high": 0.0,
        "knn_cross_source_fraction_ci_low": 0.5,
        "knn_cross_source_fraction_ci_high": 0.5,
    },
    rtol=1e-6,
    atol=1e-7,
)
```

Cover allclose true/false, centroid CI zero/nonzero width, kNN CI zero/nonzero width, and generated/not-applicable identity fields.

**Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'deficit or identity'`

Expected: missing helpers.

**Step 3: Implement pure helpers**

Add:

```python
IDENTITY_RTOL = 1e-6
IDENTITY_ATOL = 1e-7

def measured_power_deficit_depth(power_ratio: ArrayLike) -> float: ...
def interval_has_zero_width(low: float, high: float) -> bool: ...
def classify_transform_identity(original, transformed, metric, *, rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL) -> dict[str, bool | str]: ...
def generated_identity_diagnostics() -> dict[str, bool | str]: ...
```

Compute the deficit over finite bins only and reject an all-nonfinite vector. Compare arrays after explicit float32 conversion. Identity is true only when allclose and both CI widths are machine-scale zero.

**Step 4: Run tests and commit**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'deficit or identity'`

Expected: PASS.

Commit:

```bash
git add simdiff_eval/probe_c4_umap.py tests/test_probe_c4_umap.py
git commit -m "feat: classify identity-equivalent C4 controls"
```

### Task 2: Add exact and empirical mixing baselines

**Files:**
- Modify: `simdiff_eval/probe_c4_umap.py`
- Modify: `tests/test_probe_c4_umap.py`

**Step 1: Write failing baseline tests**

Test the exact finite-population formula for equal and unequal groups, including rejection of zero/negative counts. Build a four-simulation fixture and verify `deterministic_balanced_real_split()` returns equal A/B counts within every simulation, has stable membership for seed 123, changes membership with a different seed, and that `real_split_mixing_baseline()` returns the same metric/CI keys as `compare_source_to_reference()`.

**Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'perfect_mixing or real_split or baseline'`

Expected: missing baseline APIs.

**Step 3: Implement baseline helpers**

Add:

```python
def perfect_mixing_expectation(source_count: int, reference_count: int) -> float: ...
def deterministic_balanced_real_split(sim_index: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]: ...
def real_split_mixing_baseline(features, sim_index, *, k: int, n_boot: int, seed: int) -> dict[str, Any]: ...
```

Within each sorted simulation, shuffle row indices with one seeded generator, assign the first half to A and the second half to B, and reject odd or undersized blocks. Record the seed, rule, and per-simulation membership indices. Reuse `compare_source_to_reference()` unchanged for the statistic and simulation-block bootstrap.

**Step 4: Run tests and commit**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'perfect_mixing or real_split or baseline or knn'`

Expected: PASS.

Commit:

```bash
git add simdiff_eval/probe_c4_umap.py tests/test_probe_c4_umap.py
git commit -m "feat: add C4 mixing reference baselines"
```

### Task 3: Separate complete, headline, identity, and visual-only outputs

**Files:**
- Modify: `scripts/evaluate_probe_c4_umap.py`
- Modify: `tests/test_probe_c4_umap.py`

**Step 1: Write failing orchestration tests**

Extract and test small table-building functions with synthetic metadata/features/arrays. Assert every complete metric row repeats `gaussian_sigma_pixels`, `measured_power_deficit_depth`, `source_count`, `reference_count`, `perfect_mixing_expectation`, `real_split_mixing_baseline`, its CI, and all identity fields. Assert headline rows have only `feature_space == "frozen_standardized_vgg_mlp_input_1024d"` and `transform_is_identity == False`. Assert generated rows are not applicable, and the identity JSON/CSV says exactly `transform had no effect at this N` with run, N, source, transform, sigma, deficit, and diagnostics.

**Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'headline or complete or identity_report or fitted_parameter'`

Expected: evaluator still merges 1024-D and 2-D metrics into one table.

**Step 3: Refactor metric construction**

Retain `real_images`, measured arrays, and Gaussian arrays in a mapping keyed by run/source until metrics are classified. Compute quantitative metrics only in standardized 1024-D space. Build:

- `c4_feature_metrics_complete.csv/json` with all source rows and identity diagnostics;
- `c4_feature_metrics_headline.csv/json` excluding identity rows;
- `c4_transform_identity_report.csv/json` with only transform arms and the explicit no-effect message;
- `umap_layout_separation_visual_only.csv/json` if retaining 2-D diagnostics.

Do not use `centroid_distance` as a UMAP column name; use `umap_layout_separation_visual_only`. Repeat the saved sigma and computed deficit on every corresponding run row without refitting either transform.

**Step 4: Update the frozen analysis contract**

Change `_analysis_config()` to `c4_frozen_vgg_umap_seed123_v3`. Record identity tolerances and criteria, headline space, split seed/membership rule, exact perfect-mixing formula, and the statement that UMAP is visualization-only.

**Step 5: Run tests and commit**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'headline or complete or identity or fitted_parameter or analysis'`

Expected: PASS.

Commit:

```bash
git add scripts/evaluate_probe_c4_umap.py tests/test_probe_c4_umap.py
git commit -m "feat: separate C4 v3 scientific outputs"
```

### Task 4: Make UMAP warnings explicit and other warnings visible

**Files:**
- Modify: `scripts/evaluate_probe_c4_umap.py`
- Modify: `tests/test_probe_c4_umap.py`

**Step 1: Write failing warning tests**

Use a fake reducer that emits `UserWarning("Graph is not fully connected")` plus a second unexpected warning. Assert the graph warning is captured into:

```json
{
  "umap_graph_not_fully_connected": true,
  "warnings": [{"category": "UserWarning", "message": "Graph is not fully connected ..."}]
}
```

and the unexpected warning is re-emitted through `warnings.warn_explicit`. Test a connected fit records false and an empty warning list.

**Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'umap_warning or disconnected'`

Expected: current fit has no warning contract.

**Step 3: Implement targeted capture**

Add `fit_umap_with_connectivity_report(reducer, standardized)`. Use `warnings.catch_warnings(record=True)` with `simplefilter("always")`; retain only messages containing `Graph is not fully connected` in the report and re-emit every other warning with its category, filename, and line number.

**Step 4: Run tests and commit**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'umap_warning or disconnected or pooled'`

Expected: PASS.

Commit:

```bash
git add scripts/evaluate_probe_c4_umap.py tests/test_probe_c4_umap.py
git commit -m "fix: expose C4 UMAP connectivity warnings"
```

### Task 5: Add narrow warning compatibility and a mixing-baseline figure

**Files:**
- Modify: `scripts/evaluate_probe_c4_umap.py`
- Modify: `tests/test_probe_c4_umap.py`

**Step 1: Write failing compatibility tests**

Assert the warning filter matches only PyTorch's `TypedStorage is deprecated` message. Simulate the known threadpoolctl callback failure and prove the wrapper returns a recorded compatibility diagnostic; simulate an unrelated callback exception and prove it propagates. Test a synthetic headline table produces a mixing plot with observed confidence intervals, perfect-expectation line, empirical-baseline line, and a distinct identity marker only in the complete-diagnostics plot.

**Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'typedstorage or threadpool or mixing_plot'`

Expected: missing compatibility and plot helpers.

**Step 3: Implement narrow handling and plot**

Install one message/category-scoped `warnings.filterwarnings()` for TypedStorage at entry. Wrap only threadpool information collection, catch only the known callback signature/`NoneType` failure, and record its type/message; re-raise all other exceptions. Add `c4_knn_mixing_baselines.png` showing observed CI bars plus the exact and empirical horizontal references, labeled in plain language.

**Step 4: Run tests and commit**

Run: `pytest -q tests/test_probe_c4_umap.py -k 'typedstorage or threadpool or mixing_plot'`

Expected: PASS.

Commit:

```bash
git add scripts/evaluate_probe_c4_umap.py tests/test_probe_c4_umap.py
git commit -m "feat: add auditable C4 warning and mixing plot"
```

### Task 6: Finalize C4 through the shared terminal-report lifecycle and v3 wrapper

**Files:**
- Modify: `scripts/evaluate_probe_c4_umap.py`
- Modify: `scripts/slurm/probe_c4_frozen_vgg_umap.sbatch`
- Modify: `tests/test_probe_c4_umap.py`
- Modify: `tests/test_terminal_report_policy.py`

**Step 1: Write failing wrapper/lifecycle tests**

Assert the evaluator writes an INCOMPLETE `manifest.json` in staging, atomically publishes the entire staging directory, and never embeds terminal PASS. Assert the sbatch output directory is exactly `c4_frozen_vgg_umap_seed123_v3`, refuses overwrite, installs an EXIT trap, finalizes PASS only after every expected artifact exists, and finalizes FAILED on normal errors. Assert policy audit recognizes this producer as compliant.

**Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_probe_c4_umap.py tests/test_terminal_report_policy.py`

Expected: v2 path and literal PASS are still present.

**Step 3: Implement lifecycle and wrapper checks**

The evaluator creates its manifest with `start_report()` and all scientific/provenance fields, then atomically renames staging to the final v3 directory. The sbatch wrapper owns finalization against `${OUTPUT_DIR}/manifest.json`, including job ID and exit code. It verifies nonempty complete/headline/identity/visual/provenance/config/NPZ/reducer/figure artifacts before PASS.

**Step 4: Run focused tests and commit**

Run: `pytest -q tests/test_probe_c4_umap.py tests/test_terminal_reports.py tests/test_terminal_report_policy.py`

Expected: PASS.

Commit:

```bash
git add scripts/evaluate_probe_c4_umap.py scripts/slurm/probe_c4_frozen_vgg_umap.sbatch tests/test_probe_c4_umap.py tests/test_terminal_report_policy.py
git commit -m "feat: deliver fail-closed C4 v3 wrapper"
```

### Task 7: Verify C4 v3 and prepare—but do not execute—the Great Lakes payload

**Files:**
- Modify only if verification exposes a defect: files changed in Tasks 1–6

**Step 1: Run focused Python and shell checks**

Run:

```bash
python -m compileall -q simdiff_eval scripts
bash -n scripts/slurm/probe_c4_frozen_vgg_umap.sbatch
pytest -q tests/test_probe_c4_umap.py tests/test_terminal_reports.py tests/test_terminal_report_policy.py
```

Expected: all PASS.

**Step 2: Run the complete suite**

Run: `pytest -q`

Expected: all tests PASS.

**Step 3: Review scientific and path separation**

Run:

```bash
rg -n 'seed_restart|checkpoint' scripts/evaluate_probe_c4_umap.py scripts/slurm/probe_c4_frozen_vgg_umap.sbatch
rg -n 'conditional_bias_probe|c4_' scripts/slurm/*seed_restart500k*
git diff --check
```

Expected: no cross-workflow path references and no whitespace errors.

**Step 4: Prepare external-action previews**

Draft, but do not run, a distinct pinned Great Lakes `sbatch` payload targeting only `c4_frozen_vgg_umap_seed123_v3`. It must state one GPU, 80 GiB memory, two-hour wall-time ceiling, expected output size, immutable v2 preservation, and exact code/results/runtime revisions. Require a fresh `APPROVE RUN` specific to this C4 job.

