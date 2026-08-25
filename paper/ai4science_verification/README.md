# AI for Science verification workshop paper

This directory is an author-facing scaffold for an anonymous submission to the
NeurIPS 2026 workshop *Verification in the Age of AI Scientists*.

## Overleaf

1. Upload the complete `ai4science_verification` directory as a new project.
2. In Overleaf, open **Menu**, set **Main document** to `main.tex`, and recompile.
3. Keep `\usepackage[dblblindworkshop]{neurips_2026}` during double-blind review.
4. Replace the anonymous author block only for a final version and only after
   checking the workshop's current camera-ready instructions.

The scaffold intentionally contains `% AUTHOR NOTE:` comments and neutral visible
placeholders. Replace the visible placeholders with your own scientific writing.

## Local compilation

From this directory, run:

```bash
latexmk -pdf main.tex
```

Clean generated files with `latexmk -C` when needed.

## Submission constraints

- Keep the main text within **4--8 pages**; references and appendices are outside
  that target according to the workshop instructions.
- Do not add the NeurIPS checklist; this workshop does not require it.
- Keep the custom workshop notice in `main.tex`.
- Add only checked references to `references.bib`.
- Do not state a universal threshold, a verified phase law, or numerical result
  until the corresponding experiment and uncertainty analysis are complete.
