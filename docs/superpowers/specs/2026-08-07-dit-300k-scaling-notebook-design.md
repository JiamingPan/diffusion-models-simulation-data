# DiT 300k Scaling Notebook Design

## Purpose

Create a focused analysis notebook for the memorization-to-generalization transition in the DiT depth sweep. The notebook should replace the sprawling diagnostic notebook as the reader-facing comparison while preserving the older notebook as an audit trail.

The primary scientific questions are:

1. How do the PCA and SSCD novelty transitions change from DiT-L8 to L12 to L16?
2. How do those DiT transitions compare with the established UNet-64, UNet-128, and UNet-256 references?
3. Are the fresh DiT-L16 300k samples novel and physically valid across training-set sizes from `2^6` through `2^15`?
4. Do sampler step counts materially change the fresh L16 300k conclusions?

## Scope

The new notebook will be:

`notebooks/nf_generalize_fig2_dit_300k_scaling.ipynb`

It will be an unexecuted, deterministic analysis notebook intended to run on Great Lakes, where the result files are available.

The notebook will not include:

- legacy DiT-L16 200k physical-statistics panels;
- the staged 225k, 250k, 275k, and 300k legacy continuation experiment;
- claims that the mixed-budget depth comparison establishes a universal capacity law;
- silent fallback to an older sample file when a fresh 300k artifact is absent.

## Data Contract

### DiT curves

- DiT-L8: existing fixed-budget 200k PCA and SSCD tables.
- DiT-L12/base: existing fixed-budget 200k PCA and SSCD tables.
- DiT-L16: fresh independent 300k v2 PCA and SSCD tables only.
- L16 rows from the legacy 200k tables must be removed before plotting.

Every DiT plot, table, legend, and caption must identify the unequal optimizer-update budgets. The mixed comparison is useful as a current empirical diagnostic, but depth and training budget are confounded.

### UNet references

Load the established UNet-64, UNet-128, and UNet-256 PCA and SSCD full-nearest-neighbor tables from `results/nf_generalize_fig2/tables`. These are historical references and must be labeled as such.

### Fresh L16 physical diagnostics

Load only the ten frozen fresh-300k-v2 manifest rows and their explicit `dpm50_fresh300k_v2` sample files. Audit each row for:

- architecture `dit_l16`;
- target of 300,000 optimizer updates;
- requested and resolved final checkpoint;
- seed 123;
- `DPMSolverMultistepScheduler`;
- 50 sampling steps;
- expected sample count;
- exact training-subset configuration used for the real reference.

The notebook must stop with an informative error if any required fresh artifact is missing or mislabeled.

## Notebook Structure

1. **TL;DR and interpretation rules**
   - State which curves use 200k and 300k updates.
   - Explain that novelty is necessary but does not imply physical validity.
2. **Input audit**
   - Show expected and observed tables, samples, checkpoints, sampler metadata, and training subsets.
3. **Generalization transition**
   - PCA and SSCD full-range curves for DiT depths.
   - A separate transition-region view.
   - UNet references shown with quiet neutral styling.
4. **Transition summary**
   - Interpolate `N50`, the training-set size where the q95 novelty score crosses 0.5.
   - Report interval/censoring status when a clean crossing is not observed.
   - Show an exploratory capacity view without a fitted universal scaling law.
5. **Fresh L16 300k validity**
   - Generated maps across all ten data sizes.
   - Generated samples versus nearest training slices.
   - One-point distributions using the exact model training subset.
   - Power-spectrum ratios with common axes.
   - Scale-resolved log-ratio heatmap and compact error summary.
   - Joint novelty-versus-physical-error view.
6. **Sampler audit**
   - Discover same-checkpoint DPM50, DPM100, DPM200, and DDPM500 files.
   - Compare only variants whose sidecar metadata resolves to the same fresh 300k checkpoint.
   - Report missing variants explicitly and provide submission commands outside the scientific conclusions.
7. **Takeaways and limitations**
   - Separate observed results from future tests.

## Figure Contract

- Use one sans-serif font family, restrained colors, and non-color distinctions.
- Use green for L8, blue for L12, and magenta for fresh L16; historical UNets use neutral grey styles.
- Never place explanatory paragraphs inside axes.
- Use separate figures when one-point and power-spectrum panels would become too dense.
- Use common axes for comparable small multiples.
- Put the update budget directly in legends or subtitles.
- Keep all data sizes from `2^6` through `2^15`; do not truncate the fresh L16 image or physical-statistics sweeps.

## Scaling Interpretation

The notebook may report observed `N50` values and compare their ordering. It must not fit or claim a clean DiT capacity scaling exponent from L8/L12 at 200k and L16 at 300k. A defensible equal-budget DiT scaling law requires L8 and L12 results at 300k or a controlled compute-matched alternative.

## Sampler Interpretation

Sampler comparisons are valid only when scheduler choice and step count are the sole changes. All variants must use the same fresh L16 300k checkpoint, seed, normalization, requested sample count, and real reference. If only DPM50 exists, the notebook should say that sampler sensitivity remains untested rather than infer that DPM50 is adequate.

## Validation

Before delivery:

- validate notebook JSON and unique cell IDs;
- compile every code cell;
- require zero saved execution outputs in the committed notebook;
- run focused structural tests for data-source exclusions, labels, figure coverage, and sampler guards;
- run the updater twice and verify idempotence;
- execute top-to-bottom on Great Lakes after the result artifacts are available;
- visually inspect the exported full-range and transition figures for overlap and clipping.
