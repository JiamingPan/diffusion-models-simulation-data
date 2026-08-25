# IAIFI Poster Scientific Copy Revision

## Goal

Revise the final 48 x 36 inch IAIFI poster without changing its established
three-column visual structure. Use Nicholas Kern's DL4Sci poster wording as an
editorial reference while limiting claims to analyses supported by the current
UNet figures.

## Scope

- Keep the current title, palette, figures, QR code, author order, and overall
  box layout.
- Do not add DiT results. The fresh DiT-L16 sweep is incomplete and should not
  be presented as a settled result.
- Do not add a posterior-coverage plot or claim posterior calibration. The
  current error bars summarize variation across generation seeds.
- Do not replace the existing figures unless a compilation or legibility issue
  requires it.

## Copy Changes

### Question and motivation

Use the direct framing from the DL4Sci poster: a trained generator acts as a
prior over cosmological fields, and a memorized empirical prior can bias
downstream inference. Describe novelty, conditioning, and physical statistics
as separate tests.

### Conditional calibration

State explicitly that each conditional generation request uses the complete
six-parameter vector
`(Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2)` from held-out test
cosmologies. Explain that the displayed panel shows only the recovered
`Omega_m` component because it is the strongest current diagnostic.

Describe the plotted 16th-84th percentile bars as seed-to-seed generation
scatter. Do not call them posterior intervals, posterior coverage, or calibrated
uncertainty. Retain the held-out real-map encoder check (`R^2 = 0.91`) as a
finite-accuracy caveat.

### Memorization and physical statistics

Clarify that nearest-neighbor novelty, one-point statistics, power spectra, and
conditional recovery answer different questions. State that every black
one-point or power-spectrum reference is calculated from the exact training
subset configured for that model, rather than the complete CAMELS collection.

### Takeaways

Replace causal or overly broad claims with three supported conclusions:

1. CAMELS HI diffusion models exhibit a sharp memorization-to-generalization
   transition as training data increase.
2. Sample novelty does not by itself establish correct physical statistics or
   conditional response.
3. Nearest-neighbor, physical-statistics, and conditional-response checks should
   be used together before treating the generator as a simulation surrogate.

## Layout and Validation

- Preserve one 48 x 36 inch landscape page.
- Keep all text readable at poster distance and avoid introducing denser copy.
- Compile the LaTeX source, inspect the log for material overfull boxes, render
  the complete PDF, and inspect the calibration, physical-statistics, and
  takeaways regions at higher resolution.
- Deliver a new print PDF while preserving the user's existing
  `poster_print.pdf`.
