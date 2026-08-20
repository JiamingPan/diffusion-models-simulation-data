# DiT 300k Scaling Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a focused Great Lakes notebook that compares DiT-L8/L12 at 200k updates with the fresh DiT-L16 v2 sweep at 300k, includes the historical UNet references, and audits novelty, optimization, generated fields, one-point statistics, power spectra, nearest neighbors, outliers, and same-checkpoint sampler sensitivity across every training size from `2^6` through `2^15`.

**Architecture:** Put reusable validation and numerical summaries in a small analysis module, generate the notebook deterministically from a builder script, and keep all result-heavy execution inside the notebook on Great Lakes. Add a separate sampler-audit Slurm path so DPM100, DPM200, and DDPM500 use the exact fresh-300k checkpoints and distinct output labels without touching the existing DPM50 files.

**Tech Stack:** Python 3.10, NumPy, pandas, Matplotlib, PyYAML, PyTorch, pytest, Jupyter notebook JSON, Slurm, existing cosmodiff sampling and PCA/SSCD analysis artifacts.

## Global Constraints

- Do not use any legacy DiT-L16 200k row, sample, checkpoint, or staged-continuation artifact in the new notebook.
- DiT-L8 and DiT-L12/base remain at 200k; fresh DiT-L16 v2 is at 300k. Label this unequal budget in every relevant figure and interpretation.
- Include all ten training sizes, `d2p06` through `d2p15`, in every fresh-L16 sweep diagnostic.
- Treat novelty and physical validity as separate quantities. Do not infer a capacity scaling exponent from the mixed-budget comparison.
- The notebook must fail loudly when a required fresh artifact, checkpoint identity, sample count, sampler field, or exact training-subset configuration is missing.
- Commit the notebook without outputs or execution counts. Execute it only on Great Lakes.
- Use `apply_patch` for manual edits and leave unrelated worktree changes untouched.

---

### Task 1: Add pure data-contract and transition helpers

**Files:**
- Create: `scripts/dit_300k_scaling_analysis.py`
- Create: `tests/test_dit_300k_scaling_analysis.py`

- [ ] Write failing unit tests for `expected_dataset_tags()` and `require_exact_dataset_sweep()` covering ten valid tags, missing tags, duplicate tags, extra tags, and non-finite `gen_gl_q95` values.
- [ ] Run `pytest -q tests/test_dit_300k_scaling_analysis.py` and confirm the tests fail because the module does not exist.
- [ ] Implement constants for the ten powers, architecture labels, update budgets, table names, sample label, seed, scheduler, and sample count.
- [ ] Implement `require_exact_dataset_sweep(table, *, arch, value_columns, context)` so every downstream plot receives one sorted row for each `d2p06` through `d2p15`.
- [ ] Write failing tests for `interpolate_n50()` covering a clean crossing, a left-censored curve, a right-censored curve, and a non-monotonic curve with multiple crossings.
- [ ] Implement log2-space interpolation that returns a structured result with `n50`, crossing interval, and censoring/status fields instead of silently choosing an ambiguous crossing.
- [ ] Write failing tests for `validate_sample_archive_metadata()` covering requested/resolved checkpoint mismatches, wrong scheduler, wrong step count, wrong seed, wrong sample count, and a valid archive metadata dictionary.
- [ ] Implement strict metadata validation and scalar normalization for NumPy archive fields.
- [ ] Run `pytest -q tests/test_dit_300k_scaling_analysis.py` and confirm all tests pass.
- [ ] Commit with message `Add DiT 300k scaling data contracts`.

### Task 2: Build the deterministic focused notebook skeleton

**Files:**
- Create: `scripts/build_dit_300k_scaling_notebook.py`
- Create: `notebooks/nf_generalize_fig2_dit_300k_scaling.ipynb`
- Create: `tests/test_dit_300k_scaling_notebook.py`

- [ ] Write failing structural tests that require the new notebook path, title, TL;DR, input audit, transition, optimization, generated-field, one-point, power-spectrum, nearest-neighbor, outlier, sampler, and limitations sections.
- [ ] Add tests that reject any of these strings in scientific data-selection code: `dpm50_cont_`, `fresh400k`, `DiT-L16 200k`, and legacy L16 fallback language.
- [ ] Add tests that require explicit labels `DiT-L8 200k`, `DiT-L12 / base 200k`, `DiT-L16 fresh 300k`, and `historical UNet reference`.
- [ ] Run `pytest -q tests/test_dit_300k_scaling_notebook.py` and confirm failure before creating the builder.
- [ ] Implement a deterministic notebook builder with stable cell IDs, `execution_count=None`, empty outputs, section metadata, and a single plotting-style setup cell.
- [ ] Add setup cells that resolve the repository root from either the repo root or `notebooks/`, import `scripts.dit_300k_scaling_analysis`, and define result, table, sample, cache, and quickcheck paths.
- [ ] Add reader-facing markdown explaining the mixed update budget and why high novelty is not evidence of physical validity.
- [ ] Generate the notebook twice and compare SHA-256 hashes to prove idempotence.
- [ ] Run the structural test and confirm it passes.
- [ ] Commit with message `Scaffold focused DiT 300k scaling notebook`.

### Task 3: Implement the audited transition and UNet comparison

**Files:**
- Modify: `scripts/build_dit_300k_scaling_notebook.py`
- Modify: `notebooks/nf_generalize_fig2_dit_300k_scaling.ipynb`
- Modify: `tests/test_dit_300k_scaling_notebook.py`

- [ ] Add failing tests requiring exact PCA/SSCD table names for the fixed-200k DiT results, fresh-300k-v2 L16 results, and historical UNet results.
- [ ] Add a failing test requiring an explicit filter that removes `dit_l16` from the historical DiT table before L8/L12 curves are assembled.
- [ ] Add input-audit cells that display path existence, row counts, architecture, dataset tag, dataset size, q95 completeness, update budget, and provenance for every curve.
- [ ] Build PCA and SSCD full-range figures using grey historical UNets, green L8, blue L12, and magenta fresh L16, with line style and marker differences in addition to color.
- [ ] Build a separate transition-region figure without cropping data from the full-range source table.
- [ ] Compute and display `N50`, interval, and censoring status for each feature and architecture.
- [ ] Add an exploratory trainable-parameter versus `N50` scatter with unconnected points and a caption stating that unequal budgets confound depth and optimization.
- [ ] Add concise markdown interpreting observed ordering only; do not fit a power law or claim universality.
- [ ] Regenerate the notebook and run `pytest -q tests/test_dit_300k_scaling_analysis.py tests/test_dit_300k_scaling_notebook.py`.
- [ ] Commit with message `Add audited DiT and UNet transition comparison`.

### Task 4: Add all-ten-size optimization diagnostics

**Files:**
- Modify: `scripts/dit_300k_scaling_analysis.py`
- Modify: `scripts/build_dit_300k_scaling_notebook.py`
- Modify: `notebooks/nf_generalize_fig2_dit_300k_scaling.ipynb`
- Modify: `tests/test_dit_300k_scaling_analysis.py`
- Modify: `tests/test_dit_300k_scaling_notebook.py`

- [ ] Write failing tests for converting epoch and batch histories to optimizer-update coordinates using `gradient_accumulation_steps` and the manifest `optimizer_steps_per_epoch`.
- [ ] Implement loss-history loading that accepts the fresh run's saved metrics format, records which series is used, and refuses to label micro-batches as optimizer updates.
- [ ] Add a ten-panel fresh-L16 loss figure, one panel per data size, with a common loss definition, consistent log scale, and the 300k endpoint marked.
- [ ] Add a second compact figure showing final loss, best loss, and median loss over the final 10% of updates versus training-set size.
- [ ] Add an audit table with run name, data size, expected/observed final update, number of epochs, loss source, and final checkpoint.
- [ ] Add markdown that separates denoising-loss convergence from novelty and physical-statistics quality.
- [ ] Extend structural tests to require all ten tags in the optimization section and reject the old three-panel `2^6,2^10,2^15` shortcut.
- [ ] Regenerate the notebook and run the two focused test files.
- [ ] Commit with message `Add full DiT-L16 optimization sweep`.

### Task 5: Add generated-field stability and nearest-training diagnostics

**Files:**
- Modify: `scripts/dit_300k_scaling_analysis.py`
- Modify: `scripts/build_dit_300k_scaling_notebook.py`
- Modify: `notebooks/nf_generalize_fig2_dit_300k_scaling.ipynb`
- Modify: `tests/test_dit_300k_scaling_analysis.py`
- Modify: `tests/test_dit_300k_scaling_notebook.py`

- [ ] Write failing tests for deterministic sample-index selection and robust common display limits from pooled fresh samples.
- [ ] Implement deterministic selection of four generated samples per data size using fixed indices, plus robust pooled color limits shared across every panel.
- [ ] Create two readable image-grid figures: `2^6` through `2^10` and `2^11` through `2^15`, each with four generated samples per training size and no cherry-picking.
- [ ] Reuse the exact configured training-subset loader to compute nearest training slices in bounded-memory chunks.
- [ ] Create nearest-training figures for all ten sizes with generated, nearest training, absolute difference, cosine similarity, and MSE.
- [ ] Add a per-sample nearest-similarity distribution figure so one displayed match is not treated as representative.
- [ ] Add explicit audit text stating that each nearest-neighbor search covers the complete configured subset for that model.
- [ ] Extend structural tests to require forty displayed generated fields, all ten nearest-training summaries, and deterministic indices.
- [ ] Regenerate the notebook and run the focused tests.
- [ ] Commit with message `Add full-sweep DiT field stability diagnostics`.

### Task 6: Add one-point, power-spectrum, and outlier distributions

**Files:**
- Modify: `scripts/dit_300k_scaling_analysis.py`
- Modify: `scripts/build_dit_300k_scaling_notebook.py`
- Modify: `notebooks/nf_generalize_fig2_dit_300k_scaling.ipynb`
- Modify: `tests/test_dit_300k_scaling_analysis.py`
- Modify: `tests/test_dit_300k_scaling_notebook.py`

- [ ] Write failing numerical tests for shared-bin one-point L1 error, per-sample power-spectrum log-ratio MAE, robust median/IQR/tail summaries, and safe handling of empty/non-positive Fourier bins.
- [ ] Implement the numerical helpers using the exact model training subset as the real reference and shared histogram bins for real and generated values.
- [ ] Add standalone one-point figures for all ten sizes with common bin edges and comparable axes.
- [ ] Add standalone mean `P_generated(k)/P_real(k)` figures for all ten sizes with a common vertical scale and an exact-agreement line at one.
- [ ] Add a scale-resolved `abs(log2(P_generated/P_real))` heatmap over training size and k bin.
- [ ] Add per-generated-sample one-point and P(k) error distributions with median, IQR, 95th percentile, and maximum markers so unstable samples remain visible.
- [ ] Add a summary table by data size containing aggregate one-point error, aggregate P(k) error, per-sample median/IQR/q95/max, and novelty q95.
- [ ] Add a joint novelty-versus-physical-error figure that highlights novel but physically inaccurate regimes without assigning a causal explanation.
- [ ] Cache expensive per-sample FFT summaries under `results/nf_generalize_fig2_dit_l16_fresh300k_v2/cache/` with a cache key containing sample checksum, real-subset configuration, and metric version.
- [ ] Extend tests to require every data size in one-point, P(k), heatmap, and per-sample sections.
- [ ] Regenerate the notebook and run focused tests.
- [ ] Commit with message `Add physical-statistics and outlier audits`.

### Task 7: Add a non-destructive same-checkpoint sampler audit path

**Files:**
- Create: `scripts/slurm/sample_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit_array.sbatch`
- Create: `scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit.sh`
- Create: `tests/test_nf_fig2_dit_l16_fresh300k_sampler_audit.py`
- Modify: `scripts/build_dit_300k_scaling_notebook.py`
- Modify: `notebooks/nf_generalize_fig2_dit_300k_scaling.ipynb`
- Modify: `scripts/slurm/README.md`

- [ ] Write failing tests that require three distinct variant labels: `dpm100_fresh300k_v2`, `dpm200_fresh300k_v2`, and `ddpm500_fresh300k_v2`.
- [ ] Add tests proving each task reads `expected_checkpoint` from the frozen fresh300k-v2 manifest and cannot write the existing `dpm50_fresh300k_v2` path.
- [ ] Implement a 30-task array mapping ten runs across DPM100, DPM200, and DDPM500, limited to two GPUs, with `OVERWRITE=0` by default and explicit scheduler/step metadata saved by `sample_cosmodiff.py`.
- [ ] Implement a submit wrapper that validates the frozen manifest, creates logs, submits the array, and prints job IDs and output labels.
- [ ] Add notebook discovery cells for DPM50, DPM100, DPM200, and DDPM500 archives.
- [ ] Require requested and resolved checkpoints, seed, sample count, normalization/config path, and real-reference identity to match before any sampler curve is drawn.
- [ ] Add same-checkpoint sampler comparisons for generated images, one-point error, and P(k) error. Show an explicit missing-variant audit instead of substituting another checkpoint.
- [ ] Add markdown with the exact Great Lakes submission command, outside the result interpretation.
- [ ] Run `pytest -q tests/test_nf_fig2_dit_l16_fresh300k_sampler_audit.py tests/test_dit_300k_scaling_notebook.py`.
- [ ] Commit with message `Add fresh DiT-L16 sampler sensitivity audit`.

### Task 8: Final notebook verification and Great Lakes handoff

**Files:**
- Modify: `README.md`
- Modify: `scripts/slurm/README.md`
- Verify: `notebooks/nf_generalize_fig2_dit_300k_scaling.ipynb`

- [ ] Add the new notebook to the README's results-notebook index and describe it as the reader-facing DiT scaling and validity analysis.
- [ ] Run the builder twice and verify byte-for-byte identical notebook output.
- [ ] Parse notebook JSON, assert unique cell IDs, compile every code cell, and assert zero saved outputs and execution counts.
- [ ] Run `pytest -q tests/test_dit_300k_scaling_analysis.py tests/test_dit_300k_scaling_notebook.py tests/test_nf_fig2_dit_l16_fresh300k_sampler_audit.py tests/test_dit_results_notebook_presentation.py`.
- [ ] Run the repository's relevant broader test selection and record any unrelated pre-existing failures separately.
- [ ] Search the generated notebook for `TODO`, `TBD`, placeholder prose, legacy L16 fallback, staged continuation labels, and unlabeled mixed-budget claims; require zero matches.
- [ ] Self-review every requirement in `docs/superpowers/specs/2026-08-07-dit-300k-scaling-notebook-design.md` against a notebook cell or test.
- [ ] Commit all final generated artifacts with message `Complete focused DiT 300k scaling notebook`.
- [ ] Push the feature branch, then fast-forward or merge it into `main` only after tests pass.
- [ ] On Great Lakes, run `bash scripts/gl_safe_pull.sh main`, execute the new notebook top to bottom, export the figures, and visually inspect full-range and transition plots for clipping, overlapping labels, missing data sizes, and inconsistent axes.
- [ ] If sampler variants are missing, submit `bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit.sh`, wait for completion, then rerun only the sampler-audit section.
