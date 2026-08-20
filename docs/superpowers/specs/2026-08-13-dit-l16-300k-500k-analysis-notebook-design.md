# DiT-L16 300k-500k Analysis Notebook Design

## Goal

Create a standalone, reader-facing notebook for the audited DiT-L16 continuation sweep. The notebook must use the corrected 300k, 340k, 380k, 420k, 460k, and 500k artifacts and must not reuse the anomalous legacy L16 200k results.

## Deliverable

The notebook is `notebooks/nf_generalize_fig2_dit_l16_300k_500k_analysis.ipynb`. It is generated deterministically by `scripts/build_dit_l16_300k_500k_analysis_notebook.py` and validated by a dedicated structural test.

## Data Contract

- Require `local/nf_generalize_fig2_dit_l16_continue500k_v2/final_audit.json` with `status == "PASS"`.
- Require exactly 10 dataset sizes, `2^6` through `2^15`.
- Require exactly six checkpoints: 300k, 340k, 380k, 420k, 460k, and 500k.
- Require 60 manifest rows, 120 corrected novelty rows, 60 physical-summary rows, 180 selected-k rows, 90 patch rows, 12 novelty tables, 60 DPM50 sample archives, and four DDPM500 controls.
- Check that every novelty table points to its checkpoint-specific sample label.
- Use the exact configured training subset as the real reference for every one-point and power-spectrum comparison.

## Notebook Structure

1. Scope, provenance, and mandatory audit.
2. Optimization histories from 300k through 500k.
3. Generated-map overview covering all dataset sizes and checkpoints.
4. PCA and SSCD generalization trajectories.
5. PCA and SSCD generalization heatmaps.
6. Threshold-crossing summary with crossing, censoring, and ambiguity reported explicitly.
7. Context comparison with historical UNet and DiT-L8/L12 references, using only audited 300k-500k L16 curves.
8. One-point error heatmap and detailed 300k/500k exact-subset figures.
9. Power-spectrum error heatmap and detailed 300k/500k ratio figures.
10. Power-spectrum mean, confidence interval, and variance at k-bins 20, 40, and 60.
11. Same-checkpoint DPM-Solver 50 versus DDPM 500 controls.
12. Patch-boundary diagnostics.
13. Nearest-training audits at `2^8` and `2^11` for 300k and 500k.
14. Joint novelty versus one-point and power-spectrum error diagnostics.
15. Evidence summary with supported conclusions and remaining limitations.

## Presentation Rules

- Core L16 analysis never displays a 200k L16 curve.
- All ten dataset sizes remain visible; no transition-region truncation.
- Use large standalone figures, stable panel dimensions, explicit subtitles, restrained colors, and non-color encodings where practical.
- Avoid declaring a universal scaling law. Threshold summaries must retain nonmonotonic and censored status.
- Keep novelty, distributional agreement, and physical fidelity as separate claims.
- The notebook is unexecuted in the repository because the audited arrays live on Great Lakes.
