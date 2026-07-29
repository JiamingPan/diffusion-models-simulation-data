import importlib.util
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "scripts" / "patch_cosmodiff_dit_class_labels.py"
STATE_PATCH_PATH = REPO_ROOT / "scripts" / "patch_cosmodiff_checkpoint_state.py"
WRAPPER_PATH = REPO_ROOT / "scripts" / "run_cosmodiff_train_with_dit_resume.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("dit_resume_patch_for_test", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_state_patch_module():
    spec = importlib.util.spec_from_file_location(
        "dit_checkpoint_state_patch_for_test", STATE_PATCH_PATH
    )
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


def make_complete_checkpoint(path: Path) -> Path:
    path.mkdir()
    for name in (
        "config.json",
        "diffusion_pytorch_model.safetensors",
        "optimizer.pkl",
        "noise_scheduler.pkl",
        "lr_scheduler.pkl",
        "random_states_0.pkl",
    ):
        (path / name).write_bytes(b"test")
    return path


class DitCheckpointResumeTests(unittest.TestCase):
    def test_checkpoint_state_patch_writes_complete_resume_state(self):
        module = load_state_patch_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            optim_path = Path(tmpdir) / "optim.py"
            optim_path.write_text(
                "from accelerate import Accelerator\n"
                "\n"
                "def train(model, optimizer, lr_scheduler, noise_scheduler, output_dir):\n"
                "    accelerator = Accelerator()\n"
                "    ckpt_save_path = output_dir\n"
                "    accelerator.save_state(ckpt_save_path)\n"
                "    accelerator.unwrap_model(model).save_pretrained(ckpt_save_path)\n"
            )

            changed = module.patch_checkpoint_state(optim_path)
            source = optim_path.read_text()

            self.assertTrue(changed)
            self.assertIn(module.MARKER, source)
            self.assertIn('os.path.join(ckpt_save_path, "optimizer.pkl")', source)
            self.assertIn('os.path.join(ckpt_save_path, "noise_scheduler.pkl")', source)
            self.assertIn('os.path.join(ckpt_save_path, "lr_scheduler.pkl")', source)
            self.assertIn('getattr(optimizer, "optimizer", optimizer)', source)
            self.assertIn('getattr(lr_scheduler, "scheduler", lr_scheduler)', source)
            self.assertFalse(module.patch_checkpoint_state(optim_path))

    def test_resume_restores_python_numpy_and_torch_rng_state(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            state = {
                "random_state": random.getstate(),
                "numpy_random_seed": np.random.get_state(),
                "torch_manual_seed": torch.get_rng_state(),
                "torch_cuda_manual_seed": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            }
            torch.save(state, checkpoint_dir / "random_states_0.pkl")

            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = float(torch.rand(1))
            for _ in range(20):
                random.random()
                np.random.random()
                torch.rand(1)

            restored = module.restore_random_states(checkpoint_dir)

            self.assertIn("python", restored)
            self.assertIn("numpy", restored)
            self.assertIn("torch_cpu", restored)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertEqual(float(torch.rand(1)), expected_torch)

    def test_resume_requires_saved_rng_state(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "random_states_0.pkl"):
                module.restore_random_states(Path(tmpdir))

    def test_resume_loader_restores_optimizer_and_scheduler_state(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            saved_model = torch.nn.Linear(2, 1)
            saved_optimizer = torch.optim.AdamW(
                saved_model.parameters(),
                lr=3.0e-4,
                weight_decay=2.0e-2,
            )
            loss = saved_model(torch.ones(1, 2)).sum()
            loss.backward()
            saved_optimizer.step()
            saved_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                saved_optimizer,
                T_max=20,
            )
            saved_optimizer.zero_grad()
            saved_model(torch.ones(1, 2)).sum().backward()
            saved_optimizer.step()
            saved_scheduler.step()

            with (checkpoint_dir / "optimizer.pkl").open("wb") as handle:
                pickle.dump(saved_optimizer, handle)
            with (checkpoint_dir / "lr_scheduler.pkl").open("wb") as handle:
                pickle.dump(saved_scheduler, handle)

            resumed_model = torch.nn.Linear(2, 1)
            optimizer, scheduler = module.restore_optimizer_and_lr_scheduler(
                resumed_model,
                checkpoint_dir,
            )

            resumed_parameters = list(resumed_model.parameters())
            optimizer_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            self.assertEqual(
                [id(parameter) for parameter in optimizer_parameters],
                [id(parameter) for parameter in resumed_parameters],
            )
            self.assertEqual(optimizer.defaults["lr"], 3.0e-4)
            self.assertEqual(optimizer.defaults["weight_decay"], 2.0e-2)
            self.assertTrue(optimizer.state)
            self.assertEqual(scheduler.last_epoch, saved_scheduler.last_epoch)
            self.assertIs(scheduler.optimizer, optimizer)

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
            make_complete_checkpoint(checkpoint_dir / "checkpoint-epoch-12499")
            make_complete_checkpoint(checkpoint_dir / "checkpoint-epoch-12791")
            target = checkpoint_dir / "checkpoint-epoch-14062"

            current, current_epoch, target_epoch = module.validate_resume_target(
                checkpoint_dir, target
            )
            self.assertEqual(current.name, "checkpoint-epoch-12791")
            self.assertEqual(current_epoch, 12_791)
            self.assertEqual(target_epoch, 14_062)

            make_complete_checkpoint(checkpoint_dir / "checkpoint-epoch-14063")
            with self.assertRaisesRegex(ValueError, "beyond exact target"):
                module.validate_resume_target(checkpoint_dir, target)

    def test_resume_ignores_a_newer_half_written_checkpoint(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            complete = make_complete_checkpoint(
                checkpoint_dir / "checkpoint-epoch-12791"
            )
            partial = checkpoint_dir / "checkpoint-epoch-13000"
            partial.mkdir()
            (partial / "config.json").write_text("{}")
            target = checkpoint_dir / "checkpoint-epoch-14062"

            current, current_epoch, _target_epoch = module.validate_resume_target(
                checkpoint_dir, target
            )

            self.assertEqual(current.resolve(), complete.resolve())
            self.assertEqual(current_epoch, 12_791)

    def test_resume_state_allows_exact_target_noop(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            target = make_complete_checkpoint(
                checkpoint_dir / "checkpoint-epoch-14062"
            )

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
            make_complete_checkpoint(base)

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
        self.assertIn("if not optimizer.state", check)
        self.assertIn("lr_scheduler.optimizer is not optimizer", check)
        self.assertIn("PASS: checkpoint resume loader reconstructed DiT", check)
        self.assertIn("PASS: optimizer moments and scheduler progress were restored", check)
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
