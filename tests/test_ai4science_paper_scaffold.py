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
        self.assertIn(
            r"\workshoptitle{Verification in the Age of AI Scientists}", main
        )
        self.assertIn(
            "Submitted to the AI for Science workshop (NeurIPS 2026).", main
        )
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

    def test_sections_have_expected_headings_and_author_prompts(self):
        expected_headings = {
            "01_introduction.tex": r"\section{Introduction}",
            "02_related_work.tex": r"\section{Related work}",
            "03_experimental_design.tex": (
                r"\section{Problem formulation and experimental design}"
            ),
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

    def test_readme_explains_overleaf_and_compilation(self):
        readme = (PAPER / "README.md").read_text()
        self.assertIn("main.tex", readme)
        self.assertIn("latexmk -pdf main.tex", readme)
        self.assertIn("4--8 pages", readme)
        self.assertIn("Do not add the NeurIPS checklist", readme)


if __name__ == "__main__":
    unittest.main()
