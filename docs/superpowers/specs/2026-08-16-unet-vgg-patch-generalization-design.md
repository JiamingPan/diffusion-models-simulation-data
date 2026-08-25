# U-Net VGG Patch-Level Generalization Design

**Date:** 2026-08-16

## Objective

Test whether the scalar memorization-to-generalization transition hides distinct
transitions for recurring local structures in CAMELS neutral-hydrogen maps.
The first pass must be falsifiable without new model training: if fixed local
feature groups all produce statistically indistinguishable generalization
curves, the feature-level hypothesis is not supported and the study stops.

This analysis uses the completed unconditional U-Net sweep at 200k optimizer
updates: U-Net-64, U-Net-128, and U-Net-256 at training-set sizes from
`2^6` through `2^15`.

## Scientific Claim Under Test

The proposed claim is not merely that CAMELS patches can be clustered. It is:

> Local feature families cross the memorization-to-generalization boundary at
> different training-set sizes, and those thresholds are related to measurable
> properties such as feature prevalence and within-feature diversity.

The null hypothesis is that all feature-conditioned curves agree within
uncertainty after accounting for feature frequency and reference-set size.

## Representation

### Patch extraction

- Divide each 128 by 128 field into a fixed non-overlapping 8 by 8 grid of
  16 by 16 patches.
- Preserve each patch's mean and standard deviation as separate amplitude
  metadata.
- Standardize each patch independently before encoding so that the primary
  clusters describe morphology rather than only brightness or density.
- Keep patch coordinates, source image identity, simulation identity, redshift,
  run identity, generated seed, and dataset-size provenance.

### Primary encoder: frozen VGG16

- Repeat the standardized single-channel patch into three channels.
- Clip standardized values to `[-4, 4]`, map them linearly to `[0, 1]`, and
  record the clipped-pixel fraction for every cache.
- Resize it to 64 by 64 with bilinear interpolation and apply ImageNet channel
  normalization.
- Use a frozen TorchVision VGG16 with the same pretrained weights family already
  used by the conditional cosmology probe.
- Read the intermediate `relu3_3` activation rather than the final VGG block.
  This level is intended to capture edges, junctions, textures, and local
  morphology while avoiding the strongest ImageNet-specific semantic bias.
- Apply spatial average and maximum pooling and concatenate the outputs,
  producing a 512-dimensional VGG descriptor.
- Fit a feature standardizer and a 64-component PCA on held-out real VGG
  descriptors only. This PCA is numerical compression for clustering and
  nearest-neighbor search; VGG16 remains the feature encoder.

The implementation must resolve `relu3_3` by a tested layer mapping rather than
depending on an undocumented magic index.

### Control representation

As a robustness check, independently fit a 32-component PCA directly on the
standardized 16 by 16 pixel patches. Results from this control must be labeled
`raw-patch PCA`; they must not be presented as the primary feature result or
mixed with the VGG cluster identities.

### Frozen feature basis

- Build deterministic, disjoint basis and evaluation pools from real CAMELS
  maps that are excluded from every evaluated training subset.
- Audit both exclusions and the basis-versus-evaluation disjointness at the
  simulation and slice level before fitting.
- Fit the VGG feature standardizer, VGG-PCA transform, and MiniBatchKMeans once.
- Use eight clusters for the first pass.
- Freeze and hash all transforms and cluster centers before evaluating any
  generated samples.
- Never refit the basis per architecture, data size, checkpoint, or generated
  sample collection.

## Reference Sampling

The largest training sets contain too many patches for a practical exact local
nearest-neighbor search. The analysis therefore uses deterministic,
image-stratified reference reservoirs:

- Use all training images when the configured training subset has at most 2,048
  images.
- Otherwise sample 2,048 training images without replacement using a fixed
  documented seed, then include all 64 patches from each selected image.
- Cache one training-reference feature set per data size because all three
  U-Net widths use the same data-selection contract at a given size.
- Run a reservoir-size sensitivity check at 1,024, 2,048, and 4,096 images for
  selected transition-region data sizes.

Every plot and table must describe the local nearest-neighbor result as
reservoir-calibrated, not exact over the full patch population.

## Metrics

Metrics are computed separately for every architecture, dataset size, and VGG
feature cluster.

### 1. Feature occupancy

Measure the fraction of real and generated patches assigned to each frozen
cluster. Report absolute occupancy error and Jensen-Shannon divergence. This
detects missing or overproduced motifs.

### 2. Feature-conditioned novelty

For each generated patch, find its most similar training-reservoir patch within
the same frozen cluster using cosine similarity in compressed VGG space.

Calibrate a cluster- and data-size-specific copy threshold using leave-one-image-
out nearest-neighbor similarities among real training patches. A generated
patch is called novel only when it is below that calibrated copy threshold.
Report the fraction novel and the full nearest-neighbor similarity distribution.
Also measure the novelty fraction of independent held-out-real patches against
the same training reservoir, providing the attainable real-data baseline.

### 3. Within-feature distribution fidelity

Compare generated and held-out-real VGG descriptors within each cluster using:

- a finite-sample-corrected Fr\'echet feature distance;
- mean and covariance discrepancies; and
- a real-versus-real split baseline at matched sample counts.

This prevents a cluster from appearing successful merely because its generated
patches are unlike the training patches.

### 4. Amplitude fidelity

Within each morphology cluster, compare the saved patch mean and standard
deviation distributions between generated and held-out-real patches. This
separates morphological fidelity from density-amplitude fidelity.

### 5. Joint acceptance

A feature cluster is considered trustworthy at a data size only if both of the
following hold:

- its novelty fraction is statistically consistent with the held-out-real
  novelty baseline under the calibrated copy threshold; and
- occupancy and within-feature distribution errors are statistically
  consistent with the matched real-versus-real baseline.

Novelty alone is never labeled generalization or success.

## Uncertainty and Transition Estimates

- Bootstrap whole images, never individual patches, because patches from one
  map are correlated.
- Use identical bootstrap image indices across clusters when possible.
- Report 95 percent confidence intervals.
- Estimate a first 50 percent novelty crossing only for descriptive purposes.
- Estimate the joint-acceptance threshold separately.
- If a curve is non-monotonic, report all crossings and do not force a sigmoid
  fit.

The first-pass feature-dependent result is considered meaningful only if at
least two sufficiently populated clusters have joint-acceptance thresholds
separated by at least two dataset-size doublings, with bootstrap intervals that
do not collapse the difference. This is a preregistered diagnostic criterion,
not a universal physical constant.

## Generated Samples and Scope

- Reuse the existing unconditional U-Net DPM-Solver 50-step sample archives.
- Include U-Net-64, U-Net-128, and U-Net-256 at every data size from `2^6`
  through `2^15`.
- Do not train new diffusion models or generate new samples in the first pass.
- Treat the three widths as separate replication families, not pooled samples.

## Artifacts

The analysis writes to a new feature-level result directory and must not
overwrite existing full-image PCA or SSCD metrics.

Required outputs:

- a manifest with every source path and artifact hash;
- the frozen VGG basis and raw-patch PCA control basis;
- cached real, training-reservoir, and generated descriptors;
- per-patch provenance tables;
- long-form cluster metric tables with bootstrap intervals;
- representative real patches nearest each cluster center;
- occupancy curves by data size;
- feature-conditioned novelty curves by data size;
- within-feature fidelity curves by data size;
- a heatmap of joint-acceptance thresholds by cluster and architecture;
- threshold versus prevalence and effective-dimension diagnostics;
- a concise executable notebook that explains the falsification result first;
  and
- a final audit JSON that fails on missing runs, moving cluster bases,
  provenance mismatches, inadequate cluster counts, or non-finite metrics.

Representative clusters remain numbered unless a human assigns a physical
label after inspecting multiple central and boundary examples. The automated
pipeline must not invent labels such as `filament` or `void`.

## Great Lakes Execution

The implementation will separate feature extraction from metric aggregation:

1. A precheck verifies manifests, sample archives, exact training-subset
   provenance, the held-out basis pool, VGG weights, and Python environment.
2. GPU jobs extract and cache frozen VGG descriptors. Arrays are throttled to at
   most two simultaneous one-GPU tasks.
3. CPU jobs fit the frozen basis, perform nearest-neighbor calculations,
   bootstrap metrics, and render tables and figures.
4. A final audit verifies completeness and frozen-basis hashes.

The submitter must use dependency type `afterok`, create log directories before
submission, print every job identifier, and provide a restart path that reuses
valid caches. No job may silently download VGG weights on a compute node.

Submitting the Great Lakes jobs is a separate external action and requires a
specific command preview and explicit `APPROVE RUN`.

## Failure Modes and Guards

- **Clusters encode brightness only:** per-patch standardization plus separate
  amplitude diagnostics.
- **ImageNet bias dominates:** intermediate VGG features, raw-patch PCA control,
  and representative-patch review.
- **Reference reservoir changes the conclusion:** reservoir-size sensitivity.
- **Frequent patches dominate uncertainty:** map-level bootstrap and per-cluster
  minimum counts.
- **Generated failures are called novel:** joint novelty and fidelity acceptance.
- **Cluster identities drift:** one frozen, hashed held-out-real basis.
- **Architecture comparisons use different data:** audit exact config-driven
  training subsets and cache references by dataset size.
- **A visually attractive taxonomy is mistaken for a result:** the notebook
  leads with threshold separation and null-test outcomes, not cluster galleries.

## Deferred Work

The first pass does not analyze when features become committed during the
denoising trajectory. That requires saved intermediate denoising states and is
added only if the data-size experiment rejects the null hypothesis. A CAMELS
self-supervised encoder or wavelet-scattering basis is also deferred until the
VGG-based falsification establishes that feature-dependent curves exist.
