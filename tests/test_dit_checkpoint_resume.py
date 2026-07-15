import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "scripts" / "patch_cosmodiff_dit_class_labels.py"
WRAPPER_PATH = REPO_ROOT / "scripts" / "run_cosmodiff_train_with_dit_resume.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("dit_resume_patch_for_test", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location("dit_resume_wrapper_for_test", WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def absolute_epoch_train(*, start_epoch, num_epochs):
    for _epoch in range(start_epoch, num_epochs):
        pass


def additional_epoch_train(*, start_epoch, num_epochs):
    for _epoch in range(start_epoch, start_epoch + num_epochs):
        pass


def indirect_additional_epoch_train(*, start_epoch, num_epochs):
    end_epoch = start_epoch + num_epochs
    for _epoch in range(start_epoch, end_epoch):
        pass


def legacy_resume_train(
    dataset,
    model=None,
    *,
    resume_from_checkpoint=None,
    num_epochs=50,
):
    start_epoch = int(str(resume_from_checkpoint).split("-")[-1]) + 1
    final_epoch = None
    for final_epoch in range(start_epoch, start_epoch + num_epochs):
        pass
    return start_epoch, num_epochs, final_epoch


class DitCheckpointResumeTests(unittest.TestCase):
    def test_epoch_argument_reproduces_and_fixes_observed_overshoot(self):
        module = load_wrapper_module()

        self.assertEqual(module.epoch_argument(12_792, 14_062, "additional"), 1_271)
        self.assertEqual(module.epoch_argument(12_792, 14_062, "absolute"), 14_063)
        self.assertEqual(12_792 + 14_063 - 1, 26_854)

    def test_epoch_semantics_are_detected_from_trainer_loop(self):
        module = load_wrapper_module()

        self.assertEqual(module.detect_epoch_semantics(absolute_epoch_train), "absolute")
        self.assertEqual(module.detect_epoch_semantics(additional_epoch_train), "additional")
        self.assertEqual(
            module.detect_epoch_semantics(indirect_additional_epoch_train), "additional"
        )
        self.assertEqual(module.detect_epoch_semantics(legacy_resume_train), "additional")

    def test_exact_target_adapter_supports_legacy_resume_checkpoint_api(self):
        module = load_wrapper_module()
        optim = SimpleNamespace(train=legacy_resume_train)

        semantics = module.install_exact_target_adapter(
            optim,
            expected_start_epoch=12_792,
            target_epoch=14_062,
        )
        result = optim.train(
            object(),
            resume_from_checkpoint="/tmp/checkpoint-epoch-12791",
            num_epochs=99_999,
        )

        self.assertEqual(semantics, "additional")
        self.assertEqual(result, (12_792, 1_271, 14_062))

    def test_resume_state_uses_latest_clean_checkpoint_and_rejects_overshoot(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            (checkpoint_dir / "checkpoint-epoch-12499").mkdir()
            (checkpoint_dir / "checkpoint-epoch-12791").mkdir()
            target = checkpoint_dir / "checkpoint-epoch-14062"

            current, current_epoch, target_epoch = module.validate_resume_target(
                checkpoint_dir, target
            )
            self.assertEqual(current.name, "checkpoint-epoch-12791")
            self.assertEqual(current_epoch, 12_791)
            self.assertEqual(target_epoch, 14_062)

            (checkpoint_dir / "checkpoint-epoch-14063").mkdir()
            with self.assertRaisesRegex(ValueError, "beyond exact target"):
                module.validate_resume_target(checkpoint_dir, target)

    def test_resume_state_allows_exact_target_noop(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            target = checkpoint_dir / "checkpoint-epoch-14062"
            target.mkdir()

            current, current_epoch, target_epoch = module.validate_resume_target(
                checkpoint_dir, target
            )
            self.assertEqual(current.resolve(), target.resolve())
            self.assertEqual(current_epoch, target_epoch)

    def test_resume_state_rejects_checkpoint_before_required_stage_start(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            base = checkpoint_dir / "checkpoint-epoch-12499"
            minimum = checkpoint_dir / "checkpoint-epoch-14062"
            target = checkpoint_dir / "checkpoint-epoch-15624"
            base.mkdir()

            with self.assertRaisesRegex(ValueError, "behind required stage start"):
                module.validate_resume_target(
                    checkpoint_dir,
                    target,
                    minimum_checkpoint=minimum,
                )

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
        self.assertIn("validate_installed_train_api", check)
        self.assertIn("bound_start_epoch", check)
        self.assertIn("detect_epoch_semantics", check)
        self.assertIn("DiTTransformer2DModel", check)
        self.assertIn("meta_parameters", check)
        self.assertIn("PASS: checkpoint resume loader reconstructed DiT", check)
        self.assertIn("PASS: installed training API supports exact-target resume", check)

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
        self.assertIn("optim.train = train_to_exact_target", wrapper)
        self.assertIn('parser.add_argument("--checkpoint-dir"', wrapper)
        self.assertIn('parser.add_argument("--minimum-checkpoint"', wrapper)
        self.assertIn('parser.add_argument("--target-checkpoint"', wrapper)
        self.assertIn("runpy.run_path", wrapper)


if __name__ == "__main__":
    unittest.main()
