# DiT Results Notebook Expansion Design

## Objective

Make `notebooks/nf_generalize_fig2_dit_results.ipynb` the single results
notebook for the DiT depth and training-data sweep. Extend its generated-map,
one-point, and power-spectrum diagnostics through `N_2D = 2^15`, preserve the
existing memorization checks requested by Nicholas Kern, and add a clearly
separated audit of the conditional calibration inputs.

The notebook must remain readable to collaborators who already understand
diffusion models. It should explain what each result establishes without
repeating introductory diffusion material.

## Scope

The implementation modifies the original tracked notebook only. It does not
create a second results notebook and does not use the untracked
`nf_generalize_fig2_dit_results_explained.ipynb` as a required artifact.

The notebook covers:

- DiT-L8, DiT-L12/base, and DiT-L16 results across `d2p06` through `d2p15`;
- generated maps and exact-subset real references;
- one-point PDFs and power-spectrum ratios;
- nearest-neighbor novelty in PCA and SSCD spaces;
- nearest-training visual comparisons;
- SSCD feature-distribution distance;
- the relationship between novelty and physical-statistics error;
- an audit of the full CAMELS conditioning vector used by the separate
  conditional calibration experiment.

It does not claim that posterior coverage or filament-level generalization has
been completed. Those require additional methodological choices and data.

## Data Selection And Provenance

Define an ordered set of all ten data tags:

`d2p06, d2p07, ..., d2p15`.

For presentation, divide them into two fixed blocks:

- low and transition data: `d2p06` through `d2p10`;
- high data: `d2p11` through `d2p15`.

Every plot must retain the architecture, run name, data tag, dataset size,
sample label, and sample path used to create it. Requested missing runs must be
reported explicitly. Plot functions must not replace a missing requested tag
with another available run.

Legacy DiT-L16, continued DiT-L16, and fresh DiT-L16 outputs are distinct
experimental series. The notebook must label them separately and never use a
fresh or continued result as an implicit replacement for the fixed-200k depth
comparison.

## Figure Design

### Generated Maps

For each DiT depth, create two figures, one for each five-tag block. Each figure
uses two rows and five columns:

- top row: generated maps;
- bottom row: real maps from the exact training subset configured for that
  model.

Use stable color limits within a figure and include the sample index and data
size in compact labels. The layout must not place titles or legends over image
panels.

### One-Point Distributions

For each DiT depth and data block, create a five-panel one-point figure. Real
and generated curves use exactly the same histogram bin edges. The black real
curve is computed from the complete training subset specified by that model's
configuration, not from the complete CAMELS archive and not from an arbitrary
cap on reference slices.

Record the real slice count and config path in an accompanying table, and
assert that the observed slice count matches the configured `N_2D`.

### Power Spectra

Separate power-spectrum figures from the one-point figures. For each depth and
data block, show generated-to-real mean `P(k)` ratios with an ideal ratio of
one. Use shared limits within a block where that does not hide a large
failure. Add a dedicated high-data zoom around unity so differences at
`2^11` through `2^15` remain visible.

### All-Data Summaries

Across all ten training sizes and all three depths, plot:

- one-point histogram L1 error;
- power-spectrum log-ratio error;
- low-, middle-, and high-`k` power-spectrum errors;
- PCA q95 novelty;
- SSCD q95 novelty.

Add a novelty-versus-physical-error scatter plot. This plot is the direct test
for samples that are unlike the training set but physically invalid. It must
not label high novelty alone as successful generalization.

## Existing Nicholas Kern Checks

Retain and clarify the existing checks:

1. For selected generated maps, display the nearest example from the complete
   configured training subset and their residual.
2. Verify that every black one-point and power-spectrum reference uses the
   exact subset used to train that model.
3. Report SSCD feature-distribution distance normalized by a real-versus-real
   finite-sample baseline.

These checks answer different questions and should remain separate:
nearest-neighbor tests detect copying, while distribution and physical
statistics test whether novel samples remain in distribution.

## Conditional Calibration Appendix

The DiT data-size sweep is unconditional. The conditional calibration analysis
must therefore appear in a clearly labeled appendix rather than being mixed
into the DiT generalization curves.

When the conditional result files are available, the appendix must:

- load held-out test-set conditioning metadata;
- verify that normalized and raw conditioning arrays have six columns, one for
  each CAMELS parameter;
- verify that the complete parameter vector is repeated consistently for each
  generated seed/sample group;
- show calibration panels for all six parameters, not only `Omega_m`;
- state that the Omega-only poster panel is a presentation subset of a
  six-dimensional conditioning experiment.

The existing 16th-to-84th-percentile bars summarize variation over generated
seeds. They are not a learned posterior. A truth-inside-seed-interval statistic
may be shown as a preparatory diagnostic, but it must be labeled explicitly as
seed-interval inclusion rather than posterior coverage.

## Interpretable Feature Follow-Up

The notebook may summarize currently available interpretable physical
components, such as low-, middle-, and high-`k` agreement and one-point tail
errors. It must not describe those summaries as filament-, void-, or
overdensity-level generalization.

True per-feature generalization and feature-selection time during denoising
remain follow-up analyses until a reproducible feature extractor and matching
criterion are specified.

## Failure Handling

- Audit all expected sample and metric files before plotting.
- Display a compact missing-artifact table and skip only affected figures.
- Raise on inconsistent dataset sizes, conditioning dimensions, or sample
  provenance.
- Avoid silent fallbacks to another architecture, data size, checkpoint, or
  sample label.
- Keep expensive real-data loading streamed or batched so the notebook does
  not repeat the earlier memory failure.

## Verification

- Add notebook-source tests for all ten tags, the two presentation blocks,
  exact-subset references, the full six-parameter conditioning audit, and the
  posterior-coverage disclaimer.
- Compile every code cell.
- Validate notebook JSON and cell ordering.
- Run focused tests locally without executing Great Lakes-only analyses.
- Execute the complete notebook on Great Lakes, where the samples and metric
  tables exist, and inspect the resulting figures for overlapping text and
  missing series.

## Acceptance Criteria

The work is complete when the original notebook alone can be run top to bottom
on Great Lakes and produces readable low- and high-data map, one-point, and
power-spectrum figures for every available DiT depth; clearly reports missing
artifacts; preserves the existing novelty and exact-reference checks; and
verifies that the conditional calibration experiment uses the complete
six-parameter test-set vector without overstating posterior coverage.
