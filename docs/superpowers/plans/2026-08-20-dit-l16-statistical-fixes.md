# DiT-L16 Statistical Analysis Fixes

## Scope

Repair the audited DiT-L16 300k-to-500k physics analysis without retraining,
regenerating samples, or changing legacy `simdiff_eval` defaults. Each task is
implemented and committed independently after its focused tests pass.

## Task 1: Nyquist-Limited Power Spectra

- Add opt-in `k_max` arguments to the single-image and batched power-spectrum APIs.
- Preserve the exact legacy path when `k_max=None`.
- Add `--k-max=64.0` to the continuation analysis and record it in the summary.
- Test legacy bin-count compatibility and the 128-pixel Nyquist bound.

## Task 2: Pixel Coverage

- Track in-range and total pixels for real and generated histograms.
- Add real/generated coverage columns to the physics summary.
- Make the final audit fail below 0.999 generated coverage with dataset/update context.
- Test fully covered and intentionally out-of-range synthetic values.

## Task 3: Shared Histogram Binning

- Define `PHYSICAL_HIST_EDGES` in `simdiff_eval` without changing existing defaults.
- Change only the continuation analysis default to 140 bins and assert shared edges.
- Record bin count and range in the summary.

## Task 4: Scalar Physics Bootstrap Intervals

- Add deterministic percentile-bootstrap helpers for histogram L1 and P(k) log-MAE.
- Bootstrap generated samples at each dataset/checkpoint pair.
- Add low/high interval columns and record bootstrap arguments.

## Task 5: Two-Sided Selected-K Bootstrap

- Retain three selected real P(k) values per streamed reference image.
- Bootstrap generated and real samples independently for ratio intervals.
- Preserve raw generated-only statistics, use the same seed, and report real SEM.
- Test deterministic behavior and a wider interval when the real set is small/noisy.

## Task 6: Real-vs-Real Physics Floor

- Split real reference images by global index parity in the same streaming pass.
- Compute histogram L1 and P(k) log-MAE between the two halves.
- Report both floor metrics and the two half counts in every summary row.

## Task 7: Crossing Extraction

- Return every log2-N crossing, crossing count, and monotonicity.
- Return NaN scalar crossings for censored or missing cases.
- Make downstream capacity calculations explicitly retain interpolated rows only.
- Test a two-crossing curve and censored behavior.

## Task 8: Honest Notebook Figures and Documentation

- Regenerate the tagged continuation block from its updater.
- Plot P(k) ratios on an unclipped logarithmic axis.
- Mark heatmap cells whose 300k and 500k confidence intervals overlap.
- Import the shared histogram edges in the notebook.
- Document which table columns and numerical grids change, including changes that
  require a Great Lakes analysis rerun to quantify exactly.

## Verification

- Run the two focused test modules after every task.
- Run the complete test suite after task 8.
- Compile every notebook code cell and run the notebook updater idempotence checks.
- Inspect the final eight-commit range and confirm no Slurm job was submitted.
