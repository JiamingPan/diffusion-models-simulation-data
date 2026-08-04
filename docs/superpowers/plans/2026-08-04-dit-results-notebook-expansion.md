# DiT Results Notebook Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the original DiT results notebook a readable, provenance-safe analysis of all ten training-set sizes, with high-data physical-statistics plots and a six-parameter conditional-input audit.

**Architecture:** Extend the source-based notebook tests first, then update the existing deterministic notebook rewriter to patch `nf_generalize_fig2_dit_results.ipynb` in place. Keep data loading and plotting inside the notebook because Great Lakes owns the result files, while local tests validate notebook source, JSON, cell compilation, and labeling.

**Tech Stack:** Python 3.10+, Jupyter notebook JSON, NumPy, pandas, Matplotlib, pytest, existing `simdiff_eval` utilities.

## Global Constraints

- Modify `notebooks/nf_generalize_fig2_dit_results.ipynb` as the only required notebook artifact.
- Cover `d2p06` through `d2p15` for DiT-L8, DiT-L12/base, and DiT-L16 when artifacts exist.
- Split detailed figures into `d2p06`-`d2p10` and `d2p11`-`d2p15` blocks.
- Use each model's complete configured training subset for black reference curves.
- Never substitute another tag, architecture, checkpoint series, or sample label for missing requested data.
- Keep unconditional DiT results separate from the conditional calibration appendix.
- Call seed-quantile inclusion a seed-interval diagnostic, not posterior coverage.
- Do not claim filament-, void-, or denoising-time feature generalization without a defined extractor.

---

### Task 1: Lock Notebook Requirements With Failing Tests

**Files:**
- Modify: `tests/test_dit_results_notebook_presentation.py`

**Interfaces:**
- Consumes: notebook source returned by `notebook_source()`.
- Produces: source-level acceptance tests for tag coverage, block layout, physical-error summaries, and conditional-input auditing.

- [ ] **Step 1: Add failing coverage and provenance tests**

```python
def test_dit_notebook_covers_all_training_sizes_in_readable_blocks():
    source = notebook_source()
    assert "ALL_DATA_TAGS = [f'd2p{i:02d}' for i in range(6, 16)]" in source
    assert "LOW_DATA_TAGS = ALL_DATA_TAGS[:5]" in source
    assert "HIGH_DATA_TAGS = ALL_DATA_TAGS[5:]" in source
    assert "requested tags are missing" in source
    assert "max_count=5" not in source


def test_dit_notebook_summarizes_physical_error_for_all_depths():
    source = notebook_source()
    assert "onepoint_hist_l1" in source
    assert "pk_log_ratio_mae" in source
    assert "pk_low_log_ratio_mae" in source
    assert "pk_mid_log_ratio_mae" in source
    assert "pk_high_log_ratio_mae" in source
    assert "novelty versus physical-statistics error" in source


def test_conditional_appendix_audits_full_parameter_vector_without_claiming_coverage():
    source = notebook_source()
    assert "Conditional Calibration Input Audit" in source
    assert "expected_parameter_count = 6" in source
    assert "theta_norm_repeated" in source
    assert "theta_raw" in source
    assert "seed-interval inclusion; not posterior coverage" in source
```

- [ ] **Step 2: Run tests and verify the new assertions fail**

Run: `pytest -q tests/test_dit_results_notebook_presentation.py`

Expected: failures for missing all-tag blocks, summary metrics, and conditional appendix.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_dit_results_notebook_presentation.py
git commit -m "Test expanded DiT notebook coverage"
```

### Task 2: Expand Maps And Physical-Statistics Figures

**Files:**
- Modify: `scripts/update_dit_notebook_readability.py`
- Modify: `notebooks/nf_generalize_fig2_dit_results.ipynb`
- Test: `tests/test_dit_results_notebook_presentation.py`

**Interfaces:**
- Consumes: `loaded`, `choose_bundles(tags, arch)`, `field_histogram`, `batch_power_spectra`, and exact-subset reference metadata already built by the notebook.
- Produces: `ALL_DATA_TAGS`, `LOW_DATA_TAGS`, `HIGH_DATA_TAGS`, block-specific map/PDF/P(k) figures, and a per-run physical-statistics table.

- [ ] **Step 1: Add deterministic tag blocks and strict selection**

Insert into the notebook setup cell:

```python
ALL_DATA_TAGS = [f'd2p{i:02d}' for i in range(6, 16)]
LOW_DATA_TAGS = ALL_DATA_TAGS[:5]
HIGH_DATA_TAGS = ALL_DATA_TAGS[5:]
DATA_TAG_BLOCKS = {
    'low_transition': LOW_DATA_TAGS,
    'high_data': HIGH_DATA_TAGS,
}
```

Change `choose_bundles` so it preserves requested order, reports all missing
`(arch, tag)` pairs using the phrase `requested tags are missing`, and returns
only exact matches. Remove `max_count` from calls and signatures.

- [ ] **Step 2: Refactor generated-map figures into five-column blocks**

Implement:

```python
def plot_dit_image_grid(
    *, sample_index: int, tags: list[str], arch: str, block_name: str
) -> Path | None:
    ...
```

The function must render generated maps above exact-subset real references,
use common limits per figure, and save
`nf_generalize_fig2_{arch}_{block_name}_generated_image_grid.png`.

- [ ] **Step 3: Refactor one-point and power-spectrum figures**

Implement:

```python
def plot_dit_onepoint_pk(
    *, tags: list[str], arch: str, block_name: str
) -> dict[str, Path] | None:
    ...
```

Use shared histogram edges for real and generated values. Save separate
`..._{block_name}_onepoint.png`, `..._{block_name}_pk_ratio.png`, and for the
high-data block `..._high_data_pk_ratio_zoom.png`.

- [ ] **Step 4: Run the updater against the original notebook**

Run:

```bash
python scripts/update_dit_notebook_readability.py \
  --input notebooks/nf_generalize_fig2_dit_results.ipynb \
  --output notebooks/nf_generalize_fig2_dit_results.ipynb
```

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_dit_results_notebook_presentation.py`

Expected: all existing tests pass; the Task 1 coverage test passes.

- [ ] **Step 6: Commit maps and physical-statistics expansion**

```bash
git add scripts/update_dit_notebook_readability.py notebooks/nf_generalize_fig2_dit_results.ipynb tests/test_dit_results_notebook_presentation.py
git commit -m "Expand DiT diagnostics through high data"
```

### Task 3: Add All-Data Physical-Error And Novelty Summaries

**Files:**
- Modify: `scripts/update_dit_notebook_readability.py`
- Modify: `notebooks/nf_generalize_fig2_dit_results.ipynb`
- Test: `tests/test_dit_results_notebook_presentation.py`

**Interfaces:**
- Consumes: per-run histograms and power-spectrum ratios produced by `plot_dit_onepoint_pk`, plus PCA and SSCD q95 metric tables already loaded by the notebook.
- Produces: `dit_physical_summary_df` with one-point and scale-banded P(k) errors, error-vs-data-size plots, and novelty-vs-error plots.

- [ ] **Step 1: Compute per-run physical errors**

For each exact `(arch, dataset_tag)` bundle, add rows with:

```python
{
    'onepoint_hist_l1': float(np.sum(np.abs(gen_hist - real_hist) * np.diff(edges))),
    'pk_log_ratio_mae': float(np.nanmean(np.abs(np.log10(valid_ratio)))),
    'pk_low_log_ratio_mae': band_error(valid_ratio, 0.0, 1.0 / 3.0),
    'pk_mid_log_ratio_mae': band_error(valid_ratio, 1.0 / 3.0, 2.0 / 3.0),
    'pk_high_log_ratio_mae': band_error(valid_ratio, 2.0 / 3.0, 1.0),
}
```

`band_error` divides finite k bins by index into three non-overlapping bands
and returns mean absolute log10 ratio.

- [ ] **Step 2: Plot errors across all ten training sizes**

Create one figure for one-point and total P(k) error, and one for low/mid/high-k
P(k) error. Use architecture color and line style consistently, log base-two
x labels, and no fitted scaling relationship.

- [ ] **Step 3: Join novelty with physical error**

Join exact `(arch, dataset_tag)` rows to PCA and SSCD q95 values, then plot
`q95 novelty score` against `pk_log_ratio_mae`. Label the desirable region as
high novelty and low error without labeling every high-novelty point as
generalizing.

- [ ] **Step 4: Reapply updater and run focused tests**

Run:

```bash
python scripts/update_dit_notebook_readability.py --input notebooks/nf_generalize_fig2_dit_results.ipynb --output notebooks/nf_generalize_fig2_dit_results.ipynb
pytest -q tests/test_dit_results_notebook_presentation.py
```

Expected: all presentation tests pass, including all summary-column assertions.

- [ ] **Step 5: Commit all-data summaries**

```bash
git add scripts/update_dit_notebook_readability.py notebooks/nf_generalize_fig2_dit_results.ipynb tests/test_dit_results_notebook_presentation.py
git commit -m "Relate DiT novelty to physical error"
```

### Task 4: Add Full-Vector Conditional Calibration Audit

**Files:**
- Modify: `scripts/update_dit_notebook_readability.py`
- Modify: `notebooks/nf_generalize_fig2_dit_results.ipynb`
- Test: `tests/test_dit_results_notebook_presentation.py`

**Interfaces:**
- Consumes: conditional probe NPZ metadata containing `theta_norm_repeated` and `theta_raw`, held-out parameter files, and optional six-parameter calibration CSVs.
- Produces: a separate conditional appendix with provenance assertions, six-panel calibration display when data exist, and seed-interval inclusion diagnostics.

- [ ] **Step 1: Insert appendix markdown that separates experiments**

State that the DiT sweep above is unconditional and that the appendix audits a
separate conditional UNet experiment. State that the Omega-only poster panel is
a presentation subset of the full six-parameter conditioning experiment.

- [ ] **Step 2: Audit full-vector metadata**

Implement code equivalent to:

```python
expected_parameter_count = 6
assert theta_norm_repeated.ndim == 2
assert theta_raw.ndim == 2
assert theta_norm_repeated.shape[1] == expected_parameter_count
assert theta_raw.shape[1] == expected_parameter_count
```

For each held-out condition, verify that repeated rows are identical within
that sample group and display file path, shape, and pass/fail status. Missing
conditional files should produce an explicit audit table rather than stop the
unconditional notebook.

- [ ] **Step 3: Display six-parameter calibration and seed-interval inclusion**

When the calibration table is present, render a 2x3 truth-versus-prediction
figure using all CAMELS parameters. Add a compact fraction of truths lying in
the generated-seed 16th-to-84th-percentile interval and label it exactly:
`seed-interval inclusion; not posterior coverage`.

- [ ] **Step 4: Reapply updater and run focused tests**

Run:

```bash
python scripts/update_dit_notebook_readability.py --input notebooks/nf_generalize_fig2_dit_results.ipynb --output notebooks/nf_generalize_fig2_dit_results.ipynb
pytest -q tests/test_dit_results_notebook_presentation.py
```

Expected: all tests pass, including full-vector and coverage-label assertions.

- [ ] **Step 5: Commit conditional audit**

```bash
git add scripts/update_dit_notebook_readability.py notebooks/nf_generalize_fig2_dit_results.ipynb tests/test_dit_results_notebook_presentation.py
git commit -m "Audit full conditional parameter vectors"
```

### Task 5: Validate And Prepare Great Lakes Handoff

**Files:**
- Modify if required: `notebooks/nf_generalize_fig2_dit_results.ipynb`
- Test: `tests/test_dit_results_notebook_presentation.py`

**Interfaces:**
- Consumes: final notebook JSON.
- Produces: a structurally valid, compilable notebook and exact Great Lakes update/execution instructions.

- [ ] **Step 1: Validate notebook JSON and compile every code cell**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('notebooks/nf_generalize_fig2_dit_results.ipynb')
notebook = json.loads(path.read_text())
seen = set()
for index, cell in enumerate(notebook['cells']):
    cell_id = cell.get('id')
    assert cell_id and cell_id not in seen, (index, cell_id)
    seen.add(cell_id)
    if cell.get('cell_type') == 'code':
        compile(''.join(cell.get('source', [])), f'{path}:cell-{index}', 'exec')
print(f'validated {len(notebook["cells"])} cells')
PY
```

Expected: one validation line and no exception.

- [ ] **Step 2: Run focused and related tests**

Run:

```bash
pytest -q \
  tests/test_dit_results_notebook_presentation.py \
  tests/test_real_reference_loading.py \
  tests/test_feature_distribution_distance.py \
  tests/test_nearest_training_matches.py
```

Expected: all tests pass.

- [ ] **Step 3: Inspect the final diff for accidental output or unrelated changes**

Run:

```bash
git diff --check
git diff --stat -- notebooks/nf_generalize_fig2_dit_results.ipynb scripts/update_dit_notebook_readability.py tests/test_dit_results_notebook_presentation.py
```

- [ ] **Step 4: Execute on Great Lakes**

After pulling the resulting commit on Great Lakes, run:

```bash
cd /home/jiamingp/diffusion_models_repo
jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/nf_generalize_fig2_dit_results.ipynb
```

Inspect both five-column map blocks, PDF panels, full-scale and zoomed P(k)
figures, missing-artifact table, and six-parameter appendix before sharing.

- [ ] **Step 5: Commit any validation-only adjustments**

```bash
git add notebooks/nf_generalize_fig2_dit_results.ipynb scripts/update_dit_notebook_readability.py tests/test_dit_results_notebook_presentation.py
git commit -m "Finalize expanded DiT results notebook"
```
