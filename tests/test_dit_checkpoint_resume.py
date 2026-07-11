import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "scripts" / "patch_cosmodiff_dit_class_labels.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("dit_resume_patch_for_test", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DitCheckpointResumeTests(unittest.TestCase):
    def test_checkpoint_loader_uses_saved_diffusers_class(self):
        module = load_patch_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            utils_path = Path(tmpdir) / "utils.py"
            utils_path.write_text(
                "def load_checkpoint(ckpt_path):\n"
                "    model = AutoModel.from_pretrained(ckpt_path)\n"
                "    return model\n"
            )

            changed = module.patch_checkpoint_loader(utils_path)
            source = utils_path.read_text()

            self.assertTrue(changed)
            self.assertIn(module.CHECKPOINT_MARKER, source)
            self.assertIn('model_config.get("_class_name")', source)
            self.assertIn("getattr(diffusers, class_name)", source)
            self.assertIn("meta_parameters", source)
            self.assertNotIn("model = AutoModel.from_pretrained(ckpt_path)", source)
            self.assertFalse(module.patch_checkpoint_loader(utils_path))

    def test_continuation_submission_is_gated_by_resume_precheck(self):
        submit = (
            REPO_ROOT / "scripts" / "slurm" / "submit_nf_generalize_fig2_dit_l16_continue.sh"
        ).read_text()
        self.assertIn("precheck_nf_generalize_fig2_dit_l16_resume.sbatch", submit)
        self.assertIn("previous_job=${resume_precheck}", submit)

    def test_resume_precheck_loads_actual_checkpoint_and_rejects_meta_parameters(self):
        check = (REPO_ROOT / "scripts" / "check_nf_generalize_fig2_dit_resume.py").read_text()
        self.assertIn("load_checkpoint_preserving_class", check)
        self.assertIn("DiTTransformer2DModel", check)
        self.assertIn("meta_parameters", check)
        self.assertIn("PASS: checkpoint resume loader reconstructed DiT", check)

    def test_training_uses_in_process_dit_resume_wrapper(self):
        train = (
            REPO_ROOT / "scripts" / "slurm" / "train_nf_generalize_fig2_dit_l16_continue_array.sbatch"
        ).read_text()
        wrapper = (REPO_ROOT / "scripts" / "run_cosmodiff_train_with_dit_resume.py").read_text()

        self.assertIn("run_cosmodiff_train_with_dit_resume.py", train)
        self.assertNotIn('"${PYTHON_BIN}" scripts/train_cosmodiff.py', train)
        self.assertIn("load_checkpoint_preserving_class", wrapper)
        self.assertIn('model_config.get("_class_name")', wrapper)
        self.assertIn("utils.load_checkpoint = load_checkpoint_preserving_class", wrapper)
        self.assertIn("runpy.run_path", wrapper)


if __name__ == "__main__":
    unittest.main()
