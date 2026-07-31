import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_sample_module():
    path = REPO_ROOT / "scripts" / "sample_cosmodiff.py"
    spec = importlib.util.spec_from_file_location("sample_cosmodiff_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContinuationSamplingGuardTests(unittest.TestCase):
    def test_npz_output_records_resolved_checkpoint(self):
        module = load_sample_module()
        samples = np.arange(12, dtype=np.float32).reshape(3, 1, 2, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "samples.npz"
            module.save_sample_output(
                output,
                samples,
                requested_checkpoint=Path("/runs/example_checkpoints"),
                resolved_checkpoint=Path("/runs/example_checkpoints/checkpoint-epoch-18750"),
                config_path=Path("/configs/example.yaml"),
                scheduler_name="DPMSolverMultistepScheduler",
                num_steps=50,
                seed=123,
            )

            with np.load(output) as data:
                np.testing.assert_array_equal(data["samples"], samples)
                self.assertEqual(str(data["resolved_checkpoint"].item()), "/runs/example_checkpoints/checkpoint-epoch-18750")
                self.assertEqual(str(data["requested_checkpoint"].item()), "/runs/example_checkpoints")
                self.assertEqual(str(data["scheduler"].item()), "DPMSolverMultistepScheduler")
                self.assertEqual(int(data["num_steps"].item()), 50)
                self.assertEqual(int(data["seed"].item()), 123)

    def test_one_sampler_uses_explicit_checkpoint_override(self):
        text = (REPO_ROOT / "scripts" / "slurm" / "sample_nf_generalize_fig2_one_sampler.sbatch").read_text()
        self.assertIn("CHECKPOINT_PATH=${CHECKPOINT_PATH:-${CHECKPOINT_DIR}}", text)
        self.assertEqual(text.count('--checkpoint "${CHECKPOINT_PATH}"'), 2)

    def test_continuation_saves_every_epoch_by_default(self):
        text = (REPO_ROOT / "scripts" / "slurm" / "train_nf_generalize_fig2_continue400k.sbatch").read_text()
        self.assertIn("CHECKPOINT_EVERY_N_EPOCHS=${CHECKPOINT_EVERY_N_EPOCHS:-1}", text)
        self.assertIn('--checkpoint-every-n-epochs "${CHECKPOINT_EVERY_N_EPOCHS}"', text)

    def test_notebook_rejects_identical_samples(self):
        notebook_path = REPO_ROOT / "notebooks" / "nf_generalize_fig2_partial_quickcheck.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn("INVALID COMPARISON", source)
        self.assertIn("np.array_equal", source)
        self.assertIn("resolved_checkpoint", source)

    def test_notebook_defaults_to_the_verified_continuation_checkpoint(self):
        notebook_path = REPO_ROOT / "notebooks" / "nf_generalize_fig2_partial_quickcheck.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn("dpm50,dpm50_cont_epoch18768", source)
        self.assertNotIn("200k_vs_400k_fidelity", source)

    def test_controlled_fidelity_uses_complete_training_reference(self):
        notebook_path = REPO_ROOT / "notebooks" / "nf_generalize_fig2_partial_quickcheck.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )

        self.assertIn("max_raw_samples=None", source)
        self.assertIn("FULL TRAINING REFERENCE MISMATCH", source)
        self.assertIn("real_reference_kind", source)

    def test_all_quickcheck_real_curves_use_full_normalization_reference(self):
        for notebook_name in (
            "nf_generalize_fig2_partial_quickcheck.ipynb",
            "nf_generalize_fig2_dit_results.ipynb",
        ):
            notebook_path = REPO_ROOT / "notebooks" / notebook_name
            notebook = json.loads(notebook_path.read_text())
            source = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
            )

            self.assertIn("load_real_reference_from_config", source, notebook_name)
            self.assertIn("MAX_REAL_REFERENCE_SLICES", source, notebook_name)
            self.assertIn("normalized from complete configured training set", source, notebook_name)
            self.assertNotIn("max_raw_samples=raw_cap", source, notebook_name)

    def test_unet_one_point_plot_discloses_model_subset_and_plotting_cap(self):
        notebook_path = REPO_ROOT / "notebooks" / "nf_generalize_fig2_partial_quickcheck.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )

        self.assertIn("configured_training_reference_info", source)
        self.assertIn("reference_matches_manifest", source)
        self.assertIn("model training subset", source)
        self.assertIn("n_real_used_for_plot", source)
        self.assertIn("not all available CAMELS maps", source)


if __name__ == "__main__":
    unittest.main()
