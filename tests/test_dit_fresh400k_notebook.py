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

    def test_fresh_400k_section_uses_frozen_manifest_and_all_ten_sizes(self):
        self.assertIn("## Fresh DiT-L16 sweep through 400k updates", self.source)
        self.assertIn("nf_generalize_fig2_dit_l16_fresh400k", self.source)
        self.assertIn("FRESH_EXPECTED_POWERS = list(range(6, 16))", self.source)
        self.assertIn("FRESH_EXPECTED_TAGS", self.source)
        self.assertIn("fresh_400k_complete", self.source)
        self.assertIn("All ten data sizes", self.source)

    def test_fresh_400k_is_primary_300k_is_intermediate_and_200k_is_equal_budget(self):
        self.assertIn("200k is the equal-budget comparison", self.source)
        self.assertIn("300k is the intermediate L16 curve", self.source)
        self.assertIn("400k is the final L16 curve", self.source)
        self.assertIn("fresh_intermediate_updates_k = 300", self.source)
        self.assertIn("fresh_final_updates_k = 400", self.source)
        self.assertIn("fresh_equal_budget_updates_k = 200", self.source)
        self.assertIn("L16 400k", self.source)

    def test_notebook_refuses_legacy_continuation_fallback(self):
        self.assertIn("No legacy continuation fallback", self.source)
        self.assertIn("fresh_metrics_by_update", self.source)
        self.assertIn("fresh_400k_complete", self.source)
        self.assertIn("not drawing the fresh final curve", self.source)

    def test_full_range_and_transition_zoom_outputs_are_distinct(self):
        self.assertIn("fresh400k_equal_budget_200k_full.png", self.source)
        self.assertIn("fresh400k_equal_budget_200k_zoom.png", self.source)
        self.assertIn("fresh400k_intermediate_300k_full.png", self.source)
        self.assertIn("fresh400k_intermediate_300k_zoom.png", self.source)
        self.assertIn("fresh400k_final_outcome_full.png", self.source)
        self.assertIn("fresh400k_final_outcome_zoom.png", self.source)
        self.assertIn("zoom_max_power=11", self.source)

    def test_novelty_caveat_and_milestone_trajectory_are_explicit(self):
        self.assertIn("q95 novelty does not guarantee physical fidelity", self.source)
        self.assertIn("FRESH_UPDATES_K = [200, 300, 400]", self.source)
        self.assertIn("fresh400k_checkpoint_trajectories_full.png", self.source)

    def test_takeaways_make_fresh_400k_the_primary_final_l16_result(self):
        self.assertIn("Fresh 400k status:", self.source)
        self.assertIn("final longer-training L16 diagnostic", self.source)
        self.assertIn("all ten data sizes", self.source)
        self.assertIn("Legacy continuation:", self.source)
        self.assertNotIn("nf_generalize_fig2_dit_l16_fresh300k", self.source)
        self.assertNotIn(
            "The partial DiT-L16 continuation is a separate low-data diagnostic",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
