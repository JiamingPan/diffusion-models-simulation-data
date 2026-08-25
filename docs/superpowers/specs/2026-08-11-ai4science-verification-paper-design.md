# AI for Science Verification Workshop Paper Design

## Objective

Create a compilable NeurIPS 2026 workshop paper scaffold for the workshop
"Verification in the Age of AI Scientists." The paper will present an empirical
study of verification failures in scientific diffusion models: generated samples
can appear novel relative to the training set while failing distributional or
conditional scientific checks.

The scaffold must support the author's own writing. It must not contain generated
paragraph prose, invented findings, or claims that exceed the available evidence.

## Working Title

**Evaluating Scientific Generative Models Beyond Sample Novelty: Memorization,
Statistical Validity, and Conditional Consistency in Cosmological Diffusion Models**

The title remains editable and is not treated as a final submission title.

## Submission Constraints

- Use the official NeurIPS 2026 LaTeX style already supplied by the user.
- Target the workshop's 4--8 page main-text limit.
- Use anonymous authors for review.
- Replace the conference footnote with: "Submitted to the AI for Science workshop
  (NeurIPS 2026)."
- Do not include the NeurIPS paper checklist because the workshop explicitly says
  it is not required.
- Keep references and appendices outside the main-text page target.

## Scientific Framing

The paper is an empirical verification study, not a claim to introduce a complete
verification standard. It distinguishes three complementary forms of evidence:

1. **Training-set reproduction:** nearest-neighbor comparisons in pixel, PCA, and
   SSCD representation spaces.
2. **Distributional validity:** agreement in one-point statistics and the power
   spectrum.
3. **Conditional consistency:** recovery of requested cosmological parameters
   using a frozen diagnostic model trained only on real data.

The central supported message is that success on sample novelty does not imply
success on scientific distributional or conditional checks. Any relationship to
an information-theoretic phase boundary is presented as interpretation and a
source of testable hypotheses, not as an already validated theoretical result.

## Main-Text Structure

### 1. Introduction

Comment prompts will cover the scientific verification problem, why sample
novelty is insufficient, the cosmological setting, and a concise list of
evidence-supported contributions.

### 2. Related Work

Subsections will cover memorization and generalization in diffusion models,
evaluation of scientific generative models, and physical or downstream
verification. Citation placeholders will be explicit and will not invent entries.

### 3. Problem Formulation and Experimental Design

Subsections will cover CAMELS neutral-hydrogen fields, the distinction between
conditional and unconditional experiments, UNet and DiT configurations,
training-set sizes and checkpoints, preprocessing, and sampling procedures.

### 4. Verification Criteria

Subsections will define training-set reproduction, distributional agreement, and
conditional consistency. Each subsection will include a comment asking the author
to state what the diagnostic can and cannot establish.

### 5. Empirical Results

Subsections will cover the data-dependent memorization-to-generalization
transition, the separation between novelty and statistical validity, conditional
parameter recovery, and architecture or optimization-time dependence. Figure
placeholders will name the required evidence without embedding preliminary plots.

### 6. Discussion

This section will address implications for verification of learned scientific
simulators, the complementary roles of sample-level, aggregate, and task-level
tests, and the relationship to phase-boundary theory. Comment prompts will require
clear separation between established findings and hypotheses.

### 7. Limitations and Future Work

Prompts will cover additional seeds, uncertainty estimates, sampler robustness,
noise- and patch-dependent tests, posterior coverage, and additional cosmological
datasets.

### 8. Conclusion

The scaffold will request a concise evidence-based conclusion without introducing
new results.

## Appendices

Appendix placeholders will cover implementation details, exact training subsets,
additional nearest-neighbor examples, supplementary physical-statistics sweeps,
and conditional-probe validation.

## File Layout

Create an isolated paper directory:

```text
paper/ai4science_verification/
  main.tex
  neurips_2026.sty
  references.bib
  sections/
    01_introduction.tex
    02_related_work.tex
    03_experimental_design.tex
    04_verification_criteria.tex
    05_results.tex
    06_discussion.tex
    07_limitations.tex
    08_conclusion.tex
  figures/
    README.md
  appendix.tex
  README.md
```

The main document will compile even while all scientific sections contain only
comment prompts and visible placeholder text. The README will explain how to set
`main.tex` as the Overleaf main document and how to upload the directory.

## Quality and Verification

- Confirm that no checklist is imported.
- Confirm that all included section files exist.
- Compile the scaffold locally when a compatible LaTeX installation is available.
- Otherwise perform a structural check of document boundaries, includes,
  bibliography wiring, and workshop footnote configuration.
- Scan for unsupported numerical claims, fabricated citations, unfinished design
  decisions, and accidental paragraph prose.

## Out of Scope

- Writing the abstract or scientific paragraphs for the author.
- Selecting final claims before the continuation experiments are analyzed.
- Uploading to Overleaf or submitting to OpenReview.
- Adding a formal verification threshold not supported by the current study.
