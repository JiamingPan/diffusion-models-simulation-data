# DiT-L16 300k-500k statistics changes

This note records the changes that affect the existing continuation physics
tables. Regenerating these tables requires only the CPU analysis; it does not
require retraining or resampling.

## Definition changes

- Power spectra now stop at the one-dimensional Nyquist limit, `k_max=64`,
  instead of extending to the two-dimensional Fourier corner near `90.51`.
  With 91 bins, the selected indices 20, 40, and 60 move from approximately
  `k={21.16, 40.84, 60.51}` to `k={15.19, 29.04, 42.88}`. Existing selected-k
  rows therefore change in both physical scale and numerical value.
- Real and generated one-point distributions now use one shared set of 140
  bins on `[-1, 1]`. The bin width changes from approximately `0.01667` for
  the previous 120-bin diagnostic to `0.01429`, a 14.3% reduction. Histogram
  L1 values can therefore change slightly even when the samples are unchanged.
- Pixel coverage inside `[-1, 1]` is recorded for the real and generated
  samples. The analysis fails when the configured coverage requirement is not
  met instead of silently renormalizing a truncated histogram.

## New uncertainty and floor columns

- The physics summary adds 95% bootstrap intervals in `hist_l1_lo`,
  `hist_l1_hi`, `pk_log10_mae_lo`, and `pk_log10_mae_hi`. The default is 2,000
  resamples with seed 123; both settings are stored in the table.
- Selected-k rows add two-sample bootstrap intervals for
  `mean(P_generated) / mean(P_real)` and include the real-reference standard
  error. These intervals resample generated and real fields independently.
- The summary adds split-real-vs-real floors for both scalar errors, together
  with `n_real_half_a` and `n_real_half_b`. These floors show the
  finite-reference variation against which generated-vs-real errors should be
  compared.

## Transition and figure interpretation

- Threshold extraction retains every crossing. Censored, nonmonotone, and
  multiply crossing curves no longer receive an endpoint as a numerical
  transition estimate. Capacity-ratio figures use only unique interpolated
  crossings.
- Power-ratio panels use a shared logarithmic y-axis that contains every value;
  large failures are no longer clipped at 3.7. Physics heatmaps mark whether
  the 300k and 500k bootstrap intervals overlap at each training size.

The exact numerical deltas in histogram L1, power-spectrum error, selected-k
ratios, and transition estimates must be read from regenerated Great Lakes
tables. They cannot be computed from this local checkout because it does not
contain the continuation sample archives.
