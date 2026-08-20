import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb"


class FreshDitL16NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text())
        cls.source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook.get("cells", [])
        )

    def test_fresh_300k_v2_section_uses_all_ten_sizes(self):
        self.assertIn("## Fresh DiT-L16 300k replacement sweep", self.source)
        self.assertIn("nf_generalize_fig2_dit_l16_fresh300k_v2", self.source)
        self.assertIn("FRESH_EXPECTED_POWERS = list(range(6, 16))", self.source)
        self.assertIn("FRESH_EXPECTED_TAGS", self.source)
        self.assertIn("fresh_300k_v2_complete", self.source)
        self.assertIn("All ten data sizes", self.source)

    def test_depth_plot_labels_unequal_training_budgets(self):
        self.assertIn("DiT-L8 and DiT-L12 use their original 200k runs", self.source)
        self.assertIn("clean 300k", self.source)
        self.assertIn("replacement sweep", self.source)
        self.assertIn("DiT-L16 300k", self.source)
        self.assertIn("DiT-L12 / base 200k", self.source)
        self.assertIn("DiT-L8 200k", self.source)

    def test_notebook_refuses_old_l16_fallback(self):
        self.assertIn("No failed continuation or old L16", self.source)
        self.assertIn("table is substituted", self.source)
        self.assertIn("fresh_300k_v2_complete", self.source)
        self.assertIn("not drawing the replacement L16", self.source)

    def test_full_range_and_transition_zoom_outputs_are_distinct(self):
        self.assertIn("fresh300k_v2_depth_comparison_full.png", self.source)
        self.assertIn("fresh300k_v2_depth_comparison_zoom.png", self.source)
        self.assertIn("zoom_max_power=11", self.source)

    def test_novelty_caveat_is_explicit(self):
        self.assertIn("q95 novelty does not guarantee physical fidelity", self.source)

    def test_takeaways_make_fresh_300k_v2_the_only_l16_depth_curve(self):
        self.assertIn("Fresh 300k v2 status:", self.source)
        self.assertIn("all ten data sizes", self.source)
        self.assertIn("The plotted L16 line comes only from the clean replacement sweep", self.source)


if __name__ == "__main__":
    unittest.main()
