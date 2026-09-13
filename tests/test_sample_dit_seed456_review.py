import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import json

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sample_dit_seed456_review as sampling


class SamplingPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.final = self.root / "final"
        self.source.mkdir()
        self.final.mkdir()
        (self.root / "config.yaml").write_text("data: {}\n")
        self.row = dict(dataset_tag="d2p08", continue_stage=5, run_name="run_resume456",
                        expected_checkpoint=str(self.final), source_checkpoint=str(self.source),
                        source_config="config.yaml", config="config.yaml")
        self.manifest = self.root / "local" / sampling.EXPERIMENT / "manifest.json"
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text(json.dumps([self.row]))
        self.review = patch.object(sampling, "review", return_value=[
            dict(dataset="d2p08", checkpoint=str(self.final), errors=[])])
        self.review.start()
        self.addCleanup(self.review.stop)

    def test_final_exact_checkpoint_and_sampler(self):
        plan = sampling.make_plan(self.root, SCRIPTS.parent, "d2p08", "500k")
        self.assertEqual(plan["checkpoint"], str(self.final))
        self.assertEqual(plan["weights"], "raw")
        self.assertIn("DPMSolverMultistepScheduler", plan["command"])
        self.assertIn("512", plan["command"])
        self.assertFalse(Path(plan["output"]).exists())

    def test_baseline_is_original_source(self):
        plan = sampling.make_plan(self.root, SCRIPTS.parent, "d2p08", "300k")
        self.assertEqual(plan["checkpoint"], str(self.source))

    def test_missing_dataset_rejected(self):
        with self.assertRaises(ValueError):
            sampling.make_plan(self.root, SCRIPTS.parent, "d2p10", "500k")

    def test_other_experiment_rejected(self):
        self.row["run_name"] = "same_seed_run"
        self.manifest.write_text(json.dumps([self.row]))
        with self.assertRaises(ValueError):
            sampling.make_plan(self.root, SCRIPTS.parent, "d2p08", "500k")

    def test_manifest_checkpoint_conflict_rejected(self):
        self.row["expected_checkpoint"] = str(self.source)
        self.manifest.write_text(json.dumps([self.row]))
        with self.assertRaises(ValueError):
            sampling.make_plan(self.root, SCRIPTS.parent, "d2p08", "500k")


if __name__ == "__main__":
    unittest.main()
