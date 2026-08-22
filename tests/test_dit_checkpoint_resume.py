import importlib.util
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


def additional_ema_train(
    *,
    start_epoch,
    num_epochs,
    ema_burn_in=1000,
):
    final_epoch = None
    for final_epoch in range(start_epoch, start_epoch + num_epochs):
        pass
    return start_epoch, num_epochs, final_epoch, ema_burn_in


def make_complete_checkpoint(path: Path) -> Path:
    path.mkdir()
    for name in (
        "config.json",
        "checkpoint_config.yaml",
        "diffusion_pytorch_model.safetensors",
        "optimizer.pkl",
        "noise_scheduler.pkl",
        "lr_scheduler.pkl",
        "random_states_0.pkl",
        "scaler.pt",
    ):
        (path / name).write_bytes(b"test")
    (path / "checkpoint_config.yaml").write_text(
        "ema_sigma_rels: null\nema_burn_in: 0\n"
    )
    return path


class DitCheckpointResumeTests(unittest.TestCase):
    def test_seed_restart_restores_every_posthoc_ema_profile_at_one_step(self):
        module = load_wrapper_module()

        class FakeKarrasEMA(torch.nn.Module):
            def __init__(self, fill: float):
                super().__init__()
                self.ema_model = torch.nn.Linear(2, 1)
                with torch.no_grad():
                    self.ema_model.weight.fill_(fill)
                    self.ema_model.bias.fill_(fill)
                self.register_buffer("initted", torch.tensor(True))
                self.register_buffer("step", torch.tensor(299_000))

        class FakePostHocEMA:
            def __init__(self, fills):
                self.ema_models = [FakeKarrasEMA(fill) for fill in fills]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir)
            (checkpoint / "checkpoint_config.yaml").write_text(
                "ema_sigma_rels: [0.02, 0.1]\nema_burn_in: 1000\n"
            )
            ema_dir = checkpoint / "ema"
            ema_dir.mkdir()
            source = FakePostHocEMA((1.25, -0.75))
            for profile_index, profile in enumerate(source.ema_models):
                checkpoint_state = {
                    key: value.to(torch.float16)
                    for key, value in profile.state_dict().items()
                }
                torch.save(
                    checkpoint_state,
                    ema_dir / f"{profile_index}.299000.pt",
                )

            resumed = FakePostHocEMA((0.0, 0.0))
            report = module.restore_posthoc_ema_state(
                resumed,
                checkpoint,
                expected_step=299_000,
                expected_sigma_rels=[0.02, 0.10],
                expected_burn_in=1_000,
            )

            self.assertEqual(report["step"], 299_000)
            self.assertEqual(report["profiles"], 2)
            for expected, profile in zip((1.25, -0.75), resumed.ema_models):
                self.assertEqual(int(profile.step.item()), 299_000)
                self.assertTrue(bool(profile.initted.item()))
                self.assertTrue(
                    torch.allclose(
                        profile.ema_model.weight,
                        torch.full_like(profile.ema_model.weight, expected),
                    )
                )

            with self.assertRaisesRegex(ValueError, "EMA sigma profiles"):
                module.restore_posthoc_ema_state(
                    FakePostHocEMA((0.0, 0.0)),
                    checkpoint,
                    expected_step=299_000,
                    expected_sigma_rels=[0.03, 0.10],
                    expected_burn_in=1_000,
                )

    def test_seed_restart_reseeds_after_load_and_audits_first_resumed_loss(self):
        module = load_wrapper_module()

        class FakeAccelerator:
            def load_state(self, checkpoint):
                random.seed(123)
                np.random.seed(123)
                torch.manual_seed(123)
                self.loaded_checkpoint = str(checkpoint)
                return "loaded"

            def backward(self, loss):
                self.backward_losses = getattr(self, "backward_losses", []) + [
                    float(loss.detach().item())
                ]

            def log(self, values, step=None):
                self.logged = (values, step)

        expected_python = random.Random(456).random()
        expected_numpy = float(np.random.RandomState(456).random_sample())
        expected_torch = float(torch.rand(1, generator=torch.Generator().manual_seed(456)))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint-epoch-9374"
            checkpoint.mkdir()
            audit_path = Path(tmpdir) / "resume_audit.json"
            module.install_seed_restart_accelerator_hooks(
                FakeAccelerator,
                checkpoint=checkpoint,
                resume_seed=456,
                source_updates=300_000,
                source_microbatches=1_200_000,
                audit_path=audit_path,
                audit_context={"run_name": "d2p08-seed456"},
            )

            accelerator = FakeAccelerator()
            self.assertEqual(accelerator.load_state(checkpoint), "loaded")
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertEqual(float(torch.rand(1)), expected_torch)

            accelerator.backward(torch.tensor(0.125))
            accelerator.backward(torch.tensor(0.500))
            accelerator.log({"train_loss": 0.25}, step=32)
            audit = __import__("json").loads(audit_path.read_text())

            self.assertEqual(audit["run_name"], "d2p08-seed456")
            self.assertEqual(audit["resume_seed"], 456)
            self.assertEqual(audit["source_updates"], 300_000)
            self.assertEqual(audit["source_microbatches"], 1_200_000)
            self.assertEqual(audit["first_resumed_optimizer_step"], 300_001)
            self.assertEqual(audit["first_resumed_microbatch_step"], 1_200_001)
            self.assertEqual(audit["first_resumed_loss"], 0.125)
            self.assertEqual(audit["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(accelerator.logged[1], 1_200_032)

    def test_resumed_checkpoint_records_original_absolute_ema_burnin(self):
        module = load_wrapper_module()

        class FakeAccelerator:
            def load_state(self, checkpoint):
                return None

            def backward(self, loss):
                return None

            def log(self, values, step=None):
                return None

            def save_state(self, output_dir):
                output = Path(output_dir)
                output.mkdir()
                (output / "checkpoint_config.yaml").write_text(
                    "ema_sigma_rels: [0.02, 0.1]\nema_burn_in: 0\n"
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint-epoch-9374"
            checkpoint.mkdir()
            output = Path(tmpdir) / "checkpoint-epoch-10624"
            module.install_seed_restart_accelerator_hooks(
                FakeAccelerator,
                checkpoint=checkpoint,
                resume_seed=456,
                source_updates=300_000,
                source_microbatches=1_200_000,
                audit_path=Path(tmpdir) / "audit.json",
                audit_context={"original_ema_burn_in": 1_000},
            )
            FakeAccelerator().save_state(output)
            config = module.yaml.safe_load(
                (output / "checkpoint_config.yaml").read_text()
            )

            self.assertEqual(config["ema_burn_in"], 1_000)
            self.assertEqual(config["resume_effective_ema_burn_in"], 0)
            self.assertTrue(config["ema_state_restored"])

    def test_later_resume_stage_keeps_rng_restored_from_previous_checkpoint(self):
        module = load_wrapper_module()

        class FakeAccelerator:
            def load_state(self, checkpoint):
                random.seed(123)

            def backward(self, loss):
                return None

            def log(self, values, step=None):
                return step

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint-epoch-10624"
            checkpoint.mkdir()
            audit_path = Path(tmpdir) / "stage2.json"
            module.install_seed_restart_accelerator_hooks(
                FakeAccelerator,
                checkpoint=checkpoint,
                resume_seed=None,
                source_updates=340_000,
                source_microbatches=1_360_000,
                audit_path=audit_path,
                audit_context={"continue_stage": 2},
            )

            accelerator = FakeAccelerator()
            accelerator.load_state(checkpoint)

            self.assertAlmostEqual(random.random(), random.Random(123).random())
            audit = __import__("json").loads(audit_path.read_text())
            self.assertEqual(audit["rng_mode"], "checkpoint_state")
            self.assertIsNone(audit["resume_seed"])

    def test_stage_one_recovery_reseeds_only_at_the_copied_300k_checkpoint(self):
        module = load_wrapper_module()
        seed_checkpoint = Path("/runs/checkpoint-epoch-9374")

        self.assertEqual(
            module.resume_seed_for_checkpoint(456, seed_checkpoint, seed_checkpoint),
            456,
        )
        self.assertIsNone(
            module.resume_seed_for_checkpoint(
                456,
                Path("/runs/checkpoint-epoch-9999"),
                seed_checkpoint,
            )
        )

    def test_completed_stage_writes_truthful_noop_audit_without_overwrite(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "noop.json"
            module.write_completed_noop_audit(
                path,
                checkpoint=Path("/runs/checkpoint-epoch-10624"),
                run_name="seedrestart",
                code_revision="abc123",
            )
            audit = __import__("json").loads(path.read_text())
            self.assertEqual(audit["status"], "already_complete_no_training")
            self.assertEqual(audit["run_name"], "seedrestart")
            with self.assertRaises(FileExistsError):
                module.write_completed_noop_audit(
                    path,
                    checkpoint=Path("/runs/checkpoint-epoch-10624"),
                    run_name="seedrestart",
                    code_revision="abc123",
                )

    def test_constant_label_is_added_in_memory_without_patching_external_repo(self):
        module = load_wrapper_module()

        class Dataset:
            def __init__(self):
                self.arrays = torch.zeros(3, 1, 2, 2)
                self.labels = None

            def __len__(self):
                return len(self.arrays)

        dataset = Dataset()
        utils = SimpleNamespace(
            parse_config_data=lambda config: {
                "data": dataset,
                "norm": None,
                "tform": None,
            }
        )
        module.install_constant_label_adapter(utils)
        output = utils.parse_config_data({"data": {"constant_label": 0}})

        self.assertTrue(torch.equal(output["data"].labels, torch.zeros(3, dtype=torch.long)))
        self.assertIsNone(output["norm"])

    def test_seed_restart_exact_target_adapter_does_not_repeat_ema_burnin(self):
        module = load_wrapper_module()
        optim = SimpleNamespace(train=additional_ema_train)

        module.install_exact_target_adapter(
            optim,
            expected_start_epoch=9_375,
            target_epoch=10_624,
            restored_ema=True,
        )
        result = optim.train(
            start_epoch=9_375,
            num_epochs=99_999,
            ema_burn_in=1_000,
        )

        self.assertEqual(result, (9_375, 1_250, 10_624, 0))

    def test_seed_restart_posthoc_factory_restores_before_training_updates(self):
        module = load_wrapper_module()

        class FakeKarrasEMA(torch.nn.Module):
            def __init__(self, fill):
                super().__init__()
                self.ema_model = torch.nn.Linear(1, 1)
                with torch.no_grad():
                    self.ema_model.weight.fill_(fill)
                    self.ema_model.bias.fill_(fill)
                self.register_buffer("initted", torch.tensor(True))
                self.register_buffer("step", torch.tensor(299_000))

        class FakePostHocEMA:
            def __init__(self, _model, **_kwargs):
                self.ema_models = [FakeKarrasEMA(0.0), FakeKarrasEMA(0.0)]

        fake_module = SimpleNamespace(PostHocEMA=FakePostHocEMA)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir)
            audit_path = Path(tmpdir) / "audit.json"
            audit_path.write_text('{"checkpoint_loaded": true}\n')
            (checkpoint / "checkpoint_config.yaml").write_text(
                "ema_sigma_rels: [0.02, 0.1]\nema_burn_in: 1000\n"
            )
            ema_dir = checkpoint / "ema"
            ema_dir.mkdir()
            for profile_index, fill in enumerate((2.0, -3.0)):
                torch.save(
                    FakeKarrasEMA(fill).state_dict(),
                    ema_dir / f"{profile_index}.299000.pt",
                )

            module.install_seed_restart_ema_factory(
                fake_module,
                checkpoint=checkpoint,
                expected_step=299_000,
                expected_sigma_rels=[0.02, 0.10],
                expected_burn_in=1_000,
                audit_path=audit_path,
            )
            restored = fake_module.PostHocEMA(object())
            audit = __import__("json").loads(audit_path.read_text())

            self.assertEqual(restored._seed_restart_report["step"], 299_000)
            self.assertEqual(audit["ema_restore"]["step"], 299_000)
            self.assertEqual(audit["ema_restore"]["profiles"], 2)
            self.assertEqual(
                float(restored.ema_models[0].ema_model.weight.item()),
                2.0,
            )
            self.assertEqual(
                float(restored.ema_models[1].ema_model.weight.item()),
                -3.0,
            )

    def test_seed_restart_contract_uses_absolute_updates_and_fixed_subset(self):
        module = load_wrapper_module()
        config = {
            "data": {
                "img_path": ["a.npy", "b.npy"],
                "n_samples": [3, 2],
                "seed": None,
                "reshape": "2d",
                "zthin": 8,
            },
            "train": {
                "gradient_accumulation_steps": 4,
                "ema_sigma_rels": [0.02, 0.10],
                "ema_update_every": 1,
                "ema_burn_in": 1_000,
            },
        }

        context = module.build_seed_restart_context(
            config,
            checkpoint_epoch=9_374,
            optimizer_steps_per_epoch=32,
            resume_ema_step=1_199_000,
            resume_seed=456,
            run_name="d2p08-seed456",
        )

        self.assertEqual(context["source_updates"], 300_000)
        self.assertEqual(context["source_microbatches"], 1_200_000)
        self.assertEqual(context["first_resumed_optimizer_step"], 300_001)
        self.assertEqual(context["first_resumed_microbatch_step"], 1_200_001)
        self.assertEqual(context["expected_ema_step"], 1_199_000)
        self.assertEqual(context["microbatches_per_optimizer_step"], 4)
        self.assertEqual(context["original_ema_burn_in"], 1_000)
        self.assertEqual(context["resume_seed"], 456)
        self.assertEqual(
            context["training_subset"]["sources"],
            [
                {
                    "img_path": "a.npy",
                    "n_samples": 3,
                    "volume_indices": [0, 1, 2],
                },
                {
                    "img_path": "b.npy",
                    "n_samples": 2,
                    "volume_indices": [0, 1],
                },
            ],
        )
        self.assertEqual(context["training_subset"]["seed"], None)
        self.assertEqual(context["training_subset"]["reshape"], "2d")
        self.assertEqual(context["training_subset"]["zthin"], 8)
        self.assertEqual(len(context["training_subset_sha256"]), 64)

    def test_partial_stage_resume_advances_the_explicit_ema_clock_by_microbatches(self):
        module = load_wrapper_module()
        self.assertEqual(
            module.ema_step_for_current_checkpoint(
                stage_start_ema_step=1_199_000,
                stage_start_epoch=9_374,
                current_epoch=9_999,
                optimizer_steps_per_epoch=32,
                microbatches_per_optimizer_step=4,
            ),
            1_279_000,
        )

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

    def test_checkpoint_state_patch_supports_accelerate_save_hook_layout(self):
        module = load_state_patch_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            optim_path = Path(tmpdir) / "optim.py"
            optim_path.write_text(
                "from accelerate import Accelerator\n"
                "\n"
                "def train(model, optimizer, lr_scheduler, noise_scheduler, output_dir):\n"
                "    accelerator = Accelerator()\n"
                "\n"
                "    def _save_model_hook(models, weights, output_dir):\n"
                "        if accelerator.is_main_process:\n"
                "            for saved_model in models:\n"
                "                saved_model.save_pretrained(output_dir)\n"
                "            weights.clear()\n"
                "\n"
                "    accelerator.register_save_state_pre_hook(_save_model_hook)\n"
                "    if accelerator.is_main_process:\n"
                "        ckpt_save_path = output_dir\n"
                "        noise_scheduler.save_pretrained(ckpt_save_path)\n"
                "        accelerator.save_state(ckpt_save_path)\n"
            )

            changed = module.patch_checkpoint_state(optim_path)
            source = optim_path.read_text()

            self.assertTrue(changed)
            self.assertIn(module.MARKER, source)
            self.assertIn('os.path.join(ckpt_save_path, "optimizer.pkl")', source)
            self.assertIn('os.path.join(ckpt_save_path, "noise_scheduler.pkl")', source)
            self.assertIn('os.path.join(ckpt_save_path, "lr_scheduler.pkl")', source)
            self.assertGreater(
                source.index(module.MARKER),
                source.index("accelerator.save_state(ckpt_save_path)"),
            )
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

    def test_resume_loader_restores_native_accelerate_optimizer_state(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            source_model = torch.nn.Linear(2, 1)
            source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=3e-4)
            source_model(torch.ones(1, 2)).sum().backward()
            source_optimizer.step()
            source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                source_optimizer, T_max=20
            )
            source_optimizer.zero_grad()
            source_model(torch.ones(1, 2)).sum().backward()
            source_optimizer.step()
            source_scheduler.step()
            torch.save(source_optimizer.state_dict(), checkpoint_dir / "optimizer.bin")
            torch.save(source_scheduler.state_dict(), checkpoint_dir / "scheduler.bin")
            (checkpoint_dir / "checkpoint_config.yaml").write_text(
                "optimizer:\n"
                "  class: torch.optim.adamw.AdamW\n"
                "lr_scheduler:\n"
                "  class: torch.optim.lr_scheduler.CosineAnnealingLR\n"
                "  kwargs:\n"
                "    T_max: 20\n"
            )

            optimizer, scheduler = module.restore_optimizer_and_lr_scheduler(
                torch.nn.Linear(2, 1), checkpoint_dir
            )

            self.assertTrue(optimizer.state)
            self.assertEqual(
                optimizer.param_groups[0]["lr"],
                source_optimizer.param_groups[0]["lr"],
            )
            self.assertEqual(scheduler.last_epoch, source_scheduler.last_epoch)

    def test_checkpoint_completeness_rejects_mixed_optimizer_scheduler_layout(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = make_complete_checkpoint(Path(tmpdir) / "checkpoint-epoch-1")
            (checkpoint / "lr_scheduler.pkl").unlink()
            (checkpoint / "scheduler.bin").write_bytes(b"test")

            self.assertIn(
                "coherent optimizer/LR scheduler state",
                module.checkpoint_missing_files(checkpoint),
            )
            self.assertFalse(module.checkpoint_is_complete(checkpoint))

    def test_native_resume_loader_rejects_corrupt_optimizer_state(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir)
            (checkpoint / "optimizer.bin").write_bytes(b"truncated")
            optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20
            )
            torch.save(scheduler.state_dict(), checkpoint / "scheduler.bin")
            (checkpoint / "checkpoint_config.yaml").write_text(
                "optimizer:\n"
                "  class: torch.optim.adamw.AdamW\n"
                "lr_scheduler:\n"
                "  class: torch.optim.lr_scheduler.CosineAnnealingLR\n"
                "  kwargs:\n"
                "    T_max: 20\n"
            )

            with self.assertRaises(Exception):
                module.restore_optimizer_and_lr_scheduler(
                    torch.nn.Linear(2, 1), checkpoint
                )

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

            utils = SimpleNamespace(find_latest_checkpoint=lambda _output: partial)
            module.install_exact_checkpoint_finder(
                utils,
                checkpoint_dir=checkpoint_dir,
                checkpoint=complete,
            )
            self.assertEqual(
                Path(utils.find_latest_checkpoint(checkpoint_dir)).resolve(),
                complete.resolve(),
            )

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

    def test_resume_refuses_an_existing_malformed_exact_target(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            make_complete_checkpoint(checkpoint_dir / "checkpoint-epoch-10000")
            target = make_complete_checkpoint(
                checkpoint_dir / "checkpoint-epoch-10624"
            )
            (target / "checkpoint_config.yaml").write_text(
                "ema_sigma_rels: [0.02, 0.1]\nema_burn_in: 1000\n"
            )
            with self.assertRaisesRegex(FileExistsError, "malformed exact target"):
                module.validate_resume_target(checkpoint_dir, target)

    def test_scientific_target_validation_requires_scaler_config_and_exact_ema(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = make_complete_checkpoint(
                Path(tmpdir) / "checkpoint-epoch-10624"
            )
            (target / "checkpoint_config.yaml").write_text(
                "ema_sigma_rels: [0.02, 0.1]\nema_burn_in: 1000\n"
            )
            ema_dir = target / "ema"
            ema_dir.mkdir()
            for profile in (0, 1):
                torch.save(
                    {
                        "step": torch.tensor(1_359_000, dtype=torch.float16),
                        "initted": torch.tensor(True),
                        "ema_model.weight": torch.ones(1),
                    },
                    ema_dir / f"{profile}.1359000.pt",
                )

            torch.save(
                {
                    "scale": 65536.0,
                    "growth_factor": 2.0,
                    "backoff_factor": 0.5,
                    "growth_interval": 2000,
                    "_growth_tracker": 0,
                },
                target / "scaler.pt",
            )

            fake_loaded = (
                torch.nn.Linear(1, 1),
                SimpleNamespace(),
                torch.optim.AdamW(torch.nn.Linear(1, 1).parameters()),
                SimpleNamespace(last_epoch=1),
                None,
            )

            with mock.patch.object(
                module,
                "load_checkpoint_preserving_class",
                return_value=fake_loaded,
            ) as loader:
                report = module.validate_scientific_checkpoint(
                    target,
                    optimizer_steps_per_epoch=32,
                    microbatches_per_optimizer_step=4,
                    expected_ema_step=1_359_000,
                    expected_ema_sigma_rels=[0.02, 0.10],
                    expected_ema_burn_in=1_000,
                )
                loader.assert_called_once_with(str(target))
            self.assertEqual(report["absolute_updates"], 340_000)
            self.assertEqual(report["absolute_microbatches"], 1_360_000)
            self.assertEqual(report["ema_step"], 1_359_000)
            self.assertEqual(report["ema_sigma_rels"], [0.02, 0.10])
            self.assertEqual(report["training_state"]["scaler_state_keys"], 5)

            with self.assertRaisesRegex(ValueError, "EMA sigma profiles"):
                module.validate_scientific_checkpoint(
                    target,
                    optimizer_steps_per_epoch=32,
                    microbatches_per_optimizer_step=4,
                    expected_ema_step=1_359_000,
                    expected_ema_sigma_rels=[0.03, 0.10],
                    expected_ema_burn_in=1_000,
                )

            (target / "scaler.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                module.validate_scientific_checkpoint(
                    target,
                    optimizer_steps_per_epoch=32,
                    microbatches_per_optimizer_step=4,
                    expected_ema_step=1_359_000,
                    expected_ema_sigma_rels=[0.02, 0.10],
                    expected_ema_burn_in=1_000,
                )

    def test_loadable_target_validation_rejects_corrupt_scaler(self):
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir)
            (checkpoint / "scaler.pt").write_bytes(b"truncated")
            with mock.patch.object(
                module,
                "load_checkpoint_preserving_class",
                return_value=(
                    torch.nn.Linear(1, 1),
                    SimpleNamespace(),
                    SimpleNamespace(state={}),
                    SimpleNamespace(last_epoch=0),
                    None,
                ),
            ):
                with self.assertRaisesRegex(Exception, "scaler"):
                    module.validate_loadable_checkpoint_state(checkpoint)

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
