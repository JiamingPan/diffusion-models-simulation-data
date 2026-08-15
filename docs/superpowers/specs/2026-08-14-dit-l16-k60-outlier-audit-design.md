# DiT-L16 k=60 Outlier Audit Design

## Objective

Determine whether the extreme low-data power-spectrum mean and variance are
caused by a small number of catastrophic generated samples or by a broadly
incorrect generated distribution.

## Selection rule

For every `(dataset_tag, updates_k)` group, compute each generated sample's

`log10(P_generated(k=60) / P_real(k=60))`.

Flag a sample only when its value differs from the group median by more than
`4.5 * 1.4826 * MAD`. This deliberately strong, two-sided criterion is applied
independently within each group. If MAD is zero or non-finite, no sample is
flagged rather than introducing an arbitrary fallback threshold.

## Required outputs

1. A group-level table containing the median, MAD, robust threshold, number of
   samples, number flagged, and number retained.
2. A sample-level table containing sample index, k=60 ratio, log ratio, robust
   score, and flag status.
3. A distribution plot showing every per-sample k=60 ratio, with flagged
   samples identified.
4. An audit gallery showing every flagged map together with its complete
   power-spectrum ratio.
5. A comparison of the original mean, sample median, and outlier-excluded mean
   at k=20, 40, and 60, plus corresponding variances and retained counts.
6. A 500k full-spectrum comparison between the original mean and the
   outlier-excluded mean.

## Interpretation constraints

- The unfiltered result remains the primary reported result.
- Exclusion is a sensitivity analysis, not data cleaning or replacement.
- Every filtered result reports `n_kept / n_total`.
- If many samples are flagged, the group is described as broadly unstable,
  not as a set of isolated outliers.
- The real reference is the exact training subset configured for each model.
