# C4 v3 and DiT-L16 Seed-Restart Safety Design

## Purpose

Deliver two independent, fail-closed workflows:

1. a corrected C4 v3 analysis of the existing conditional U-Net endpoint runs;
2. a reproducible DiT-L16 continuation of the existing 300k checkpoints to 500k with resume seed 456.

C4 is an analysis-only workflow. It reads the frozen U-Net runs
`nf_cond_bias_hi_u128_d2p07_n128_200k` and
`nf_cond_bias_hi_u128_d2p14_n16384_200k`, their saved generated samples, the
frozen VGG/MLP probe, and the saved C4 degradation parameters. It must not
write to any training checkpoint. The DiT continuation reads and writes only
the dedicated `nf_generalize_fig2_dit_l16_seed_restart500k_v1` checkpoint
tree. Their code roots, results directories, and Slurm dependencies remain
separate.

## Fixed scientific definitions

- Quantitative feature-space claims use only the frozen standardized 1024-D
  VGG/MLP input space.
- UMAP remains a pooled, frozen-input visualization. Its 2-D layout is not a
  metric space and supplies no headline centroid distance.
- `measured_power_deficit_depth` is
  `max(0, 1 - min(P_generated / P_real))` over the saved finite power-ratio
  bins. This is a reported summary of the already-saved C4 curve; it does not
  refit or redefine the transform.
- `gaussian_sigma_pixels` and `measured_power_deficit_depth` are repeated on
  every metric row for the corresponding run so each row is interpretable in
  isolation.
- The exact perfect-mixing expectation for two source populations of sizes
  `n_a` and `n_b`, with self excluded, is
  `2*n_a*n_b / ((n_a+n_b)*(n_a+n_b-1))`. For equal populations this reduces
  to `n/(2n-1)`, approximately 0.5. The expectation does not depend on `k`
  when each neighbor position is sampled from the same finite population,
  but `k`, `n_a`, and `n_b` remain recorded.

## 1. Immutable patched cosmodiff pin

Add a seed-restart pin builder and verifier rather than patching inside a
Slurm training job.

The builder will:

1. create a staging checkout at the exact base cosmodiff revision;
2. apply a declarative, ordered patch set needed by this sweep;
3. record for each patch the patch-script SHA256, whether it changed the
   checkout, and the before/after SHA256 of every target file;
4. import `cosmodiff`, `cosmodiff.optim`, `cosmodiff.utils`,
   `cosmodiff.augment`, and `cosmodiff.transform` with the same Python
   interpreter used by the jobs;
5. record the imported module paths and `cosmodiff.__version__`;
6. atomically publish the completed pin only after every check succeeds.

The pin manifest is the source of truth for a deliberately patched checkout.
Jobs will verify the base revision and every recorded final file hash instead
of requiring a clean `git status`, which cannot describe an audited patched
tree. Unexpected modified or untracked files fail verification.

The ordered seed-restart patch set is exactly:

1. `patch_cosmodiff_package_metadata.py`;
2. `patch_cosmodiff_constant_label.py`;
3. `patch_cosmodiff_dit_class_labels.py`;
4. `patch_cosmodiff_checkpoint_state.py`.

The package-metadata fallback patch is mandatory. The remaining declared
patches are run idempotently and recorded as either `applied` or
`already_supported`; this allows the pin to document capabilities already
present in the base revision without silently skipping their verification.
Patch backup sidecars are not published in the pin; the final manifest owns an
exact file inventory, and any unrecorded file causes verification to fail.

The precheck imports all required cosmodiff modules and verifies the pin
manifest before it examines manifests, checkpoints, or writes any report.

## 2. Terminal report lifecycle

Introduce one shared atomic JSON-report utility and use it for terminal
status reports. Configuration manifests, metric payloads, and ordinary
metadata files are not terminal reports and are outside this lifecycle.

A terminal report has these required fields:

- `status`: `INCOMPLETE`, `PASS`, `FAILED`, or `STALE`;
- `producer_job_id`: the Slurm job ID, or `null` outside Slurm;
- `producer_exit_code`: `null` until finalization, then an integer;
- `started_at_utc` and `finalized_at_utc`;
- `report_schema_version`.

The producer first atomically writes `INCOMPLETE`. A Slurm wrapper finalizes
the report only after all commands in that job succeed. On a normal shell
failure, an `EXIT` trap atomically writes `FAILED` with the nonzero exit code.
If the process is killed so forcefully that the trap cannot run, the report
remains `INCOMPLETE`; it can never be mistaken for `PASS`.

Downstream jobs require all of the following before consuming a report:

- `status == "PASS"`;
- `producer_exit_code == 0`;
- the expected producer job ID matches;
- the report was atomically finalized.

The seed-restart precheck will therefore create its report only as
`INCOMPLETE`, run the complete checkpoint reconstruction checks, and finalize
it last. The existing `stage1_58470485.json` will be marked `STALE` by an
explicit Great Lakes repair command after code delivery; consumers will also
reject its legacy schema even before that repair.

Repository tests will use an AST-based policy audit rather than a brittle
plain grep. Any Python or Slurm report producer that embeds terminal `PASS`
without the shared finalizer fails the test. Existing terminal-report writers
will be migrated; ordinary JSON data writers will not acquire meaningless
job-status fields.

## 3. Identity-equivalent C4 transforms

The C4 harness will preserve the transformed arrays long enough to compare
each real control directly with the original real array using explicit
float32 tolerances (`rtol=1e-6`, `atol=1e-7`). It will report these diagnostics:

- `transform_arrays_allclose`;
- `centroid_ci_zero_width`;
- `knn_ci_zero_width`;
- `transform_is_identity`;
- `identity_reason`.

For a transform arm, `transform_is_identity` is true when the arrays are
allclose and both bootstrap confidence intervals have zero width within
machine-scale tolerance. Generated samples are not transform arms and receive
`transform_is_identity = false` with a not-applicable reason.

Identity rows remain in a complete diagnostics table but are excluded from a
separate headline table. A separate JSON/CSV identity report says
`transform had no effect at this N` and includes the run name, dataset size,
source, transform name, Gaussian sigma, power-deficit depth, and both identity
diagnostics. No scientific metric is recomputed or altered.

## 4. UMAP visualization contract

UMAP is still fitted once to the same pooled standardized features and its
coordinates remain in the sample-provenance artifact and figures. The
headline quantitative table contains only the 1024-D frozen feature-space
rows.

If a 2-D layout diagnostic is retained, it is written to a separate
visual-only artifact using names such as
`umap_layout_separation_visual_only`; it is never named
`centroid_distance`, never merged into the headline metric table, and never
used by the C4 sanity-check claim.

The UMAP fit runs inside a targeted warning capture. A warning containing
`Graph is not fully connected` sets
`umap_graph_not_fully_connected = true` in the manifest and records the exact
warning category and message. Other unexpected warnings remain visible.

## 5. Mixing baselines and figures

Each headline row records:

- `source_count` and `reference_count`;
- `perfect_mixing_expectation`;
- the observed kNN cross-source fraction and its existing block-bootstrap CI;
- `real_split_mixing_baseline` and its bootstrap CI.

The empirical baseline uses one deterministic, balanced split of the shared
original-real samples within every held-out simulation, then applies the same
kNN statistic and simulation-block bootstrap used for each source comparison.
The split seed and membership rule are recorded in the analysis config. The
same baseline is repeated for every run row because the original-real
reference is shared.

A dedicated mixing plot shows observed values with intervals plus horizontal
lines for the exact perfect-mixing expectation and empirical real-vs-real
baseline. Identity rows use a distinct marker and do not enter the headline
interpretation.

## 6. Narrow warning suppression

At the C4 entry point:

- suppress only PyTorch's known `TypedStorage is deprecated` warning;
- prevent or capture the known threadpoolctl callback failure using a narrow
  compatibility wrapper around threadpool inspection, without suppressing
  unrelated exceptions or warnings;
- capture the UMAP disconnected-graph warning into the manifest as described
  above rather than suppressing it.

All other warnings and exceptions remain visible and fail normally where
appropriate.

## Output and provenance

C4 v3 writes a new, non-overwriting directory
`c4_frozen_vgg_umap_seed123_v3`. Its manifest records:

- exact code revision and clean state;
- exact U-Net source artifacts and frozen VGG/MLP artifacts;
- the saved degradation-control result revision;
- identity tolerances and flags;
- power-deficit and Gaussian parameters;
- mixing-baseline definition and seed;
- UMAP warning/connectivity state;
- terminal report lifecycle fields.

The completed v2 directory remains immutable historical evidence.

The DiT continuation keeps its existing dedicated result and checkpoint names
but changes to the verified patched cosmodiff pin and terminal report schema.
No C4 path appears in its submission chain, and no DiT checkpoint path appears
in the C4 job.

## Test and commit strategy

Implementation proceeds test first in small reviewable commits:

1. pin builder, patch manifest, import-first verifier;
2. shared terminal-report lifecycle and migration of seed precheck/report
   consumers;
3. C4 identity and fitted-parameter reporting;
4. UMAP visual-only contract and warning capture;
5. exact and empirical mixing baselines plus reference plot;
6. targeted warning-noise handling and repository-wide policy audit.

Every behavior receives a failing regression test before production changes.
Each commit runs its focused tests. Final verification runs Python compilation,
Slurm shell syntax checks, diff checks, and the complete repository test suite.

No Slurm job is submitted during implementation. After all verification is
green, delivery includes two separate, fully pinned command previews:

1. one fresh C4 v3 analysis job;
2. one DiT-L16 seed-456 continuation chain from 300k through 500k.

Each external action retains its own approval gate and cost disclosure.
