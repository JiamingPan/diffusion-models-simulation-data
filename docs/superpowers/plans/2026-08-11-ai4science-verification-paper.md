# AI for Science Verification Paper Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a compilable, anonymous NeurIPS 2026 workshop paper package that gives the author a professional structure for a scientific verification study without generating the paper's scientific prose.

**Architecture:** Keep the paper isolated under `paper/ai4science_verification/`. A small `main.tex` owns formatting and document assembly, while one file per scientific section contains visible placeholder text plus precise `% AUTHOR NOTE:` prompts. A structural test enforces the workshop-specific requirements before a full LaTeX compile verifies the package end to end.

**Tech Stack:** LaTeX, official `neurips_2026.sty`, BibTeX/natbib, Python `unittest`, `latexmk`.

## Global Constraints

- Use the official NeurIPS 2026 style file without modifying it.
- Use the double-blind workshop option and anonymous author metadata.
- Override the first-page notice with `Submitted to the AI for Science workshop (NeurIPS 2026).`
- Do not include the NeurIPS paper checklist.
- Do not write an abstract or scientific paragraph prose for the author.
- Do not invent citations, numerical results, thresholds, or claims.
- Mark writing guidance as LaTeX comments beginning with `% AUTHOR NOTE:`.
- Treat the information-theoretic phase boundary as a hypothesis and source of tests, not an established empirical result of this project.
- Do not upload to Overleaf or submit to OpenReview.

---

## Task 1: Add structural tests and the official workshop shell

**Files:**
- Create: `tests/test_ai4science_paper_scaffold.py`
- Create: `paper/ai4science_verification/main.tex`
- Copy: `paper/ai4science_verification/neurips_2026.sty`
- Create: `paper/ai4science_verification/references.bib`

- [ ] **Step 1: Write the failing structural test**

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "ai4science_verification"


class TestAI4SciencePaperScaffold(unittest.TestCase):
    def test_required_root_files_exist(self):
        for name in ("main.tex", "neurips_2026.sty", "references.bib", "README.md"):
            self.assertTrue((PAPER / name).is_file(), name)

    def test_workshop_submission_configuration(self):
        main = (PAPER / "main.tex").read_text()
        self.assertIn(r"\usepackage[dblblindworkshop]{neurips_2026}", main)
        self.assertIn(r"\workshoptitle{Verification in the Age of AI Scientists}", main)
        self.assertIn("Submitted to the AI for Science workshop (NeurIPS 2026).", main)
        self.assertIn("Anonymous Author(s)", main)
        self.assertNotIn("checklist", main.lower())

    def test_all_section_inputs_exist(self):
        main = (PAPER / "main.tex").read_text()
        expected = [
            "01_introduction",
            "02_related_work",
            "03_experimental_design",
            "04_verification_criteria",
            "05_results",
            "06_discussion",
            "07_limitations",
            "08_conclusion",
        ]
        for stem in expected:
            self.assertIn(rf"\input{{sections/{stem}}}", main)
            self.assertTrue((PAPER / "sections" / f"{stem}.tex").is_file(), stem)
        self.assertIn(r"\input{appendix}", main)
        self.assertTrue((PAPER / "appendix.tex").is_file())
```

- [ ] **Step 2: Run the test and confirm it fails because the package is absent**

Run: `python -m unittest tests.test_ai4science_paper_scaffold -v`

Expected: `FAIL` for missing paper files.

- [ ] **Step 3: Copy the official style and add the minimal document shell**

Use the locally supplied official style at:

`/Users/apple/Downloads/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.sty`

Configure `main.tex` with `dblblindworkshop`, the workshop title, anonymous author metadata, a custom notice override, empty author-written abstract placeholder, all section inputs, BibTeX, and the appendix input.

- [ ] **Step 4: Run the structural test and inspect remaining failures**

Run: `python -m unittest tests.test_ai4science_paper_scaffold -v`

Expected: root/configuration tests pass; missing section/README tests still fail.

---

## Task 2: Add the professional author-facing section scaffold

**Files:**
- Create: `paper/ai4science_verification/sections/01_introduction.tex`
- Create: `paper/ai4science_verification/sections/02_related_work.tex`
- Create: `paper/ai4science_verification/sections/03_experimental_design.tex`
- Create: `paper/ai4science_verification/sections/04_verification_criteria.tex`
- Create: `paper/ai4science_verification/sections/05_results.tex`
- Create: `paper/ai4science_verification/sections/06_discussion.tex`
- Create: `paper/ai4science_verification/sections/07_limitations.tex`
- Create: `paper/ai4science_verification/sections/08_conclusion.tex`
- Create: `paper/ai4science_verification/appendix.tex`
- Create: `paper/ai4science_verification/figures/README.md`
- Modify: `tests/test_ai4science_paper_scaffold.py`

- [ ] **Step 1: Extend the test with required section headings and prompt safeguards**

```python
    def test_sections_have_expected_headings_and_author_prompts(self):
        expected_headings = {
            "01_introduction.tex": r"\section{Introduction}",
            "02_related_work.tex": r"\section{Related work}",
            "03_experimental_design.tex": r"\section{Problem formulation and experimental design}",
            "04_verification_criteria.tex": r"\section{Verification criteria}",
            "05_results.tex": r"\section{Empirical results}",
            "06_discussion.tex": r"\section{Discussion}",
            "07_limitations.tex": r"\section{Limitations and future work}",
            "08_conclusion.tex": r"\section{Conclusion}",
        }
        for name, heading in expected_headings.items():
            text = (PAPER / "sections" / name).read_text()
            self.assertIn(heading, text)
            self.assertIn("% AUTHOR NOTE:", text)

    def test_scaffold_avoids_unfinished_submission_artifacts(self):
        tex = "\n".join(path.read_text() for path in PAPER.rglob("*.tex"))
        self.assertNotIn(r"\answerTODO", tex)
        self.assertNotIn(r"\input{checklist}", tex)
        self.assertNotIn("TODO", tex)
```

- [ ] **Step 2: Run the test and confirm section checks fail**

Run: `python -m unittest tests.test_ai4science_paper_scaffold -v`

Expected: `FAIL` because the section files do not exist yet.

- [ ] **Step 3: Create each section with concise author prompts**

The prompts must cover:

- sample-level memorization in pixel, PCA, and SSCD spaces;
- one-point and power-spectrum distributional checks;
- conditional parameter recovery with a frozen real-data diagnostic model;
- conditional versus unconditional experiments;
- UNet and DiT configurations, training sizes, checkpoints, preprocessing, and samplers;
- separation of established evidence from phase-boundary hypotheses;
- seeds, uncertainty, sampler robustness, patch/noise experiments, posterior coverage, and additional datasets.

Visible content should be limited to neutral drafting markers such as `\emph{Author-written text goes here.}` so the package compiles without pretending to contain a finished paper.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m unittest tests.test_ai4science_paper_scaffold -v`

Expected: all structural tests pass.

---

## Task 3: Add author instructions and compile the paper

**Files:**
- Create: `paper/ai4science_verification/README.md`
- Modify: `tests/test_ai4science_paper_scaffold.py`

- [ ] **Step 1: Add README-content assertions**

```python
    def test_readme_explains_overleaf_and_compilation(self):
        readme = (PAPER / "README.md").read_text()
        self.assertIn("main.tex", readme)
        self.assertIn("latexmk -pdf main.tex", readme)
        self.assertIn("4--8 pages", readme)
        self.assertIn("Do not add the NeurIPS checklist", readme)
```

- [ ] **Step 2: Run the test and confirm README checks fail**

Run: `python -m unittest tests.test_ai4science_paper_scaffold -v`

Expected: `FAIL` until the README is complete.

- [ ] **Step 3: Write concise local and Overleaf usage instructions**

Document:

- upload the entire `ai4science_verification` directory;
- set `main.tex` as Overleaf's main document;
- keep the double-blind workshop option for review;
- compile locally with `latexmk -pdf main.tex`;
- keep main text within 4--8 pages;
- do not add the checklist;
- replace anonymous authors only for an accepted/final version and only after checking workshop instructions.

- [ ] **Step 4: Run the full structural test suite**

Run: `python -m unittest tests.test_ai4science_paper_scaffold -v`

Expected: all tests pass.

- [ ] **Step 5: Compile the paper twice through `latexmk`**

Run: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

Working directory: `paper/ai4science_verification`

Expected: exit code 0 and `main.pdf` produced.

- [ ] **Step 6: Inspect the compiled PDF metadata and first page**

Confirm the PDF is readable, the title and anonymous author block render, line numbers appear, the workshop footer is correct, section placeholders do not overlap, and no checklist appears.

- [ ] **Step 7: Scan for unsupported content**

Run: `rg -n "TODO|TBD|we find|we show|our results|checklist|[0-9]+\\%" paper/ai4science_verification --glob '*.tex'`

Expected: no unsupported result prose or checklist inclusion; author notes may explicitly instruct the author what evidence to supply.
