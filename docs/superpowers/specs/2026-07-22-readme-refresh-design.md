# Public README Refresh Design

## Goal

Update `README.md` so it reflects the repository's current scope without
publishing unfinished scientific claims or internal cluster details. The text
should be direct, technical, and readable by researchers, collaborators, and
recruiters.

## Scope

The README will describe four parts of the project:

1. Training unconditional and conditional diffusion models with UNet and DiT
   backbones on two-dimensional CAMELS IllustrisTNG neutral-hydrogen fields.
2. Measuring memorization and sample novelty with nearest-neighbor and
   cross-run reproducibility diagnostics in pixel, PCA, and SSCD spaces.
3. Evaluating generated fields with one-point distributions, power spectra,
   and recovery of requested cosmological parameters using encoders trained on
   real fields.
4. Running reproducible training, checkpoint continuation, sampling, and
   analysis workflows on Great Lakes using configuration generators and Slurm
   job chains.

## Structure

The revised README will contain:

- a concise project overview;
- a section explaining the main research and engineering capabilities;
- a repository map with the important reusable modules and entry points;
- a high-level workflow for UNet/DiT sweeps and conditional calibration;
- a clear explanation of the evaluation framework;
- minimal setup guidance and pointers to representative notebooks;
- a restrained status section that identifies exploratory work without making
  final claims.

## Public-Safety Constraints

- Do not publish unpublished numerical results, run IDs, private paths,
  account names, or checkpoint locations.
- Do not present exploratory DiT-depth or capacity-scaling results as settled.
- Do not claim that one diagnostic alone proves scientific validity.
- Keep quantitative implementation facts only when they describe stable public
  interfaces, such as the available UNet and DiT templates.

## Style

- Use plain technical language and short paragraphs.
- Avoid promotional language, vague claims, and paper-abstract phrasing.
- Prefer concrete verbs such as `train`, `compare`, `measure`, and `evaluate`.
- Keep the README useful without duplicating detailed notes under `docs/`.

## Verification

After editing, check that every referenced path exists, Markdown headings and
code fences are balanced, no private filesystem paths appear, and the diff is
limited to the README plus this design record.
