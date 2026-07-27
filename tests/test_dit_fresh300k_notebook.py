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

    def test_fresh_300k_section_uses_frozen_manifest_and_all_ten_sizes(self):
        self.assertIn("## Fresh DiT-L16 sweep through 300k updates", self.source)
        self.assertIn("nf_generalize_fig2_dit_l16_fresh300k", self.source)
        self.assertIn("FRESH_EXPECTED_POWERS = list(range(6, 16))", self.source)
        self.assertIn("FRESH_EXPECTED_TAGS", self.source)
        self.assertIn("fresh_300k_complete", self.source)
        self.assertIn("All ten data sizes", self.source)

    def test_fresh_300k_is_primary_and_200k_is_only_equal_budget(self):
        self.assertIn("200k is the equal-budget comparison", self.source)
        self.assertIn("300k is the final L16 curve", self.source)
        self.assertIn("fresh_final_updates_k = 300", self.source)
        self.assertIn("fresh_equal_budget_updates_k = 200", self.source)
        self.assertIn("L16 300k", self.source)

    def test_notebook_refuses_legacy_continuation_fallback(self):
        self.assertIn("No legacy continuation fallback", self.source)
        self.assertIn("fresh_metrics_by_update", self.source)
        self.assertIn("fresh_300k_complete", self.source)
        self.assertIn("not drawing the fresh final curve", self.source)

    def test_full_range_and_transition_zoom_outputs_are_distinct(self):
        self.assertIn("fresh300k_equal_budget_200k_full.png", self.source)
        self.assertIn("fresh300k_equal_budget_200k_zoom.png", self.source)
        self.assertIn("fresh300k_final_outcome_full.png", self.source)
        self.assertIn("fresh300k_final_outcome_zoom.png", self.source)
        self.assertIn("zoom_max_power=11", self.source)

    def test_novelty_caveat_and_milestone_trajectory_are_explicit(self):
        self.assertIn("q95 novelty does not guarantee physical fidelity", self.source)
        self.assertIn("FRESH_UPDATES_K = [200, 225, 250, 275, 300]", self.source)
        self.assertIn("fresh300k_checkpoint_trajectories_full.png", self.source)

    def test_takeaways_make_fresh_300k_the_primary_final_l16_result(self):
        self.assertIn("Fresh 300k status:", self.source)
        self.assertIn("final longer-training L16 diagnostic", self.source)
        self.assertIn("all ten data sizes", self.source)
        self.assertIn("Legacy continuation:", self.source)
        self.assertNotIn(
            "The partial DiT-L16 continuation is a separate low-data diagnostic",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
