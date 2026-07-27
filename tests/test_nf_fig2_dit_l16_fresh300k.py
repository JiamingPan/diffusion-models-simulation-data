import importlib.util
import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = (
    REPO_ROOT
    / "scripts"
    / "prepare_nf_generalize_fig2_dit_l16_fresh300k_configs.py"
)
FRESH_WRAPPER_PATH = REPO_ROOT / "scripts" / "run_cosmodiff_train_fresh_seeded.py"
RESULT_AUDIT_PATH = (
    REPO_ROOT
    / "scripts"
    / "audit_nf_generalize_fig2_dit_l16_fresh300k_results.py"
)


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("fresh_l16_prepare_for_test", PREPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_fresh_wrapper_module():
    spec = importlib.util.spec_from_file_location(
        "fresh_l16_wrapper_for_test", FRESH_WRAPPER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_result_audit_module():
    spec = importlib.util.spec_from_file_location(
        "fresh_l16_result_audit_for_test", RESULT_AUDIT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FreshDitL16ManifestTests(unittest.TestCase):
    def _args(self, checkpoint_root):
        return Namespace(
            checkpoint_root=checkpoint_root,
            dataset_tag=None,
            run_name=None,
            stage=None,
            stage_updates=25_000,
            stages=12,
            safety_checkpoint_updates=5_000,
        )

    def test_all_ten_runs_train_fresh_through_300k(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = module.fresh_rows(
                self._args(Path(tmpdir) / module.SWEEP_NAME)
            )

        self.assertEqual(len(rows), 120)
        self.assertEqual(
            sorted({row["dataset_tag"] for row in rows}),
            [f"d2p{power:02d}" for power in range(6, 16)],
        )
        self.assertEqual(sorted({row["stage"] for row in rows}), list(range(1, 13)))
        self.assertEqual({row["target_total_updates"] for row in rows if row["stage"] == 12}, {300_000})
        self.assertEqual(
            {row["target_total_updates"] for row in rows if row["scientific_checkpoint"]},
            {200_000, 225_000, 250_000, 275_000, 300_000},
        )
        self.assertEqual({row["training_seed"] for row in rows}, {123})
        self.assertTrue(all(row["fresh_initialization"] for row in rows))
        self.assertTrue(all(row["arch"] == "dit_l16" for row in rows))
        self.assertTrue(all(row["run_name"].endswith("_fresh300k_seed123") for row in rows))
        self.assertTrue(
            all(
                "nf_generalize_fig2_dit_l16_fresh300k" in row["checkpoint_dir"]
                for row in rows
            )
        )
        self.assertTrue(
            all(
                "nf_generalize_fig2_dit_l16_continue" not in row["checkpoint_dir"]
                for row in rows
            )
        )

    def test_stage_arithmetic_and_scientific_labels_are_exact(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = module.fresh_rows(self._args(Path(tmpdir) / "checkpoints"))

        for row in rows:
            self.assertEqual(row["target_total_updates"], 25_000 * row["stage"])
            self.assertEqual(row["previous_target_updates"], 25_000 * (row["stage"] - 1))
            self.assertGreaterEqual(row["actual_total_updates"], row["target_total_updates"])
            self.assertLess(
                row["actual_total_updates"] - row["target_total_updates"],
                row["optimizer_steps_per_epoch"],
            )
            self.assertLessEqual(
                row["checkpoint_every_actual_updates"],
                5_000 + row["optimizer_steps_per_epoch"],
            )
            expected_scientific = row["target_total_updates"] in module.SCIENTIFIC_UPDATES
            self.assertEqual(row["scientific_checkpoint"], expected_scientific)
            if expected_scientific:
                self.assertEqual(
                    row["sample_label"],
                    f"dpm50_fresh_{row['target_total_updates'] // 1000}k",
                )
            else:
                self.assertIsNone(row["sample_label"])

    def test_configs_hold_architecture_and_training_invariants(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = module.fresh_rows(self._args(Path(tmpdir) / "checkpoints"))

        for row in (rows[0], rows[7], rows[-1]):
            config = module.build_stage_config(row)
            kwargs = config["model"]["kwargs"]
            self.assertEqual(config["model"]["class"], "DiTTransformer2DModel")
            self.assertEqual(kwargs["num_layers"], 16)
            self.assertEqual(kwargs["num_attention_heads"], 12)
            self.assertEqual(kwargs["attention_head_dim"], 64)
            self.assertEqual(kwargs["patch_size"], 8)
            self.assertEqual(config["data"]["constant_label"], 0)
            self.assertIsNone(config["data"]["seed"])
            self.assertNotIn("augmentations", config["data"])
            self.assertEqual(config["noise_scheduler"]["kwargs"]["prediction_type"], "v_prediction")
            self.assertEqual(config["optimizer"]["kwargs"]["lr"], 1.0e-4)
            self.assertEqual(config["optimizer"]["kwargs"]["weight_decay"], 1.0e-2)
            self.assertEqual(config["lr_scheduler"]["kwargs"]["T_0"], 4000)
            self.assertEqual(config["train"]["batch_size"], 2)
            self.assertEqual(config["train"]["gradient_accumulation_steps"], 4)
            self.assertEqual(config["train"]["mixed_precision"], "fp16")
            self.assertEqual(config["train"]["min_snr_gamma"], 5.0)
            self.assertEqual(config["train"]["num_epochs"], row["final_num_epochs"])

    def test_frozen_manifest_rejects_wrong_version(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            manifest = module.manifest_path(project_dir)
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps([{"manifest_version": -1}]))
            with self.assertRaisesRegex(ValueError, "manifest version"):
                module.load_existing_rows(project_dir)


class FreshDitL16WorkflowSourceTests(unittest.TestCase):
    def test_fresh_wrapper_seeds_all_cpu_rngs(self):
        module = load_fresh_wrapper_module()
        module.seed_everything(123)
        values_a = (random.random(), float(np.random.random()), float(torch.rand(1)))
        module.seed_everything(123)
        values_b = (random.random(), float(np.random.random()), float(torch.rand(1)))
        self.assertEqual(values_a, values_b)

    def test_fresh_wrapper_rejects_nonempty_checkpoint_directory(self):
        module = load_fresh_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            (checkpoint_dir / "checkpoint-epoch-0001").mkdir()
            with self.assertRaisesRegex(ValueError, "fresh checkpoint directory"):
                module.validate_fresh_checkpoint_dir(checkpoint_dir)

    def test_training_and_sampling_contract(self):
        train = (
            REPO_ROOT
            / "scripts"
            / "slurm"
            / "train_nf_generalize_fig2_dit_l16_fresh300k_array.sbatch"
        ).read_text()
        sample = (
            REPO_ROOT
            / "scripts"
            / "slurm"
            / "sample_nf_generalize_fig2_dit_l16_fresh300k_array.sbatch"
        ).read_text()

        self.assertIn("#SBATCH --time=24:00:00", train)
        self.assertIn("#SBATCH --array=0-9%2", train)
        self.assertIn("run_cosmodiff_train_fresh_seeded.py", train)
        self.assertIn("run_cosmodiff_train_with_dit_resume.py", train)
        self.assertIn("TRAIN_STAGE", train)
        self.assertIn("previous_expected_checkpoint", train)
        self.assertIn("expected_checkpoint", train)
        self.assertNotIn("nf_generalize_fig2_dit_l16_continue", train)

        self.assertIn("#SBATCH --time=04:00:00", sample)
        self.assertIn("#SBATCH --array=0-9%2", sample)
        self.assertIn("NUM_SAMPLES=${NUM_SAMPLES:-512}", sample)
        self.assertIn("SAMPLER_STEPS=${SAMPLER_STEPS:-50}", sample)
        self.assertIn("SEED=${SEED:-123}", sample)
        self.assertIn("scientific_checkpoint", sample)
        self.assertNotIn("nf_generalize_fig2_dit_l16_continue", sample)

    def test_submit_chain_reaches_300k_and_is_gated(self):
        submit = (
            REPO_ROOT
            / "scripts"
            / "slurm"
            / "submit_nf_generalize_fig2_dit_l16_fresh300k.sh"
        ).read_text()
        self.assertIn("for stage in $(seq 1 12)", submit)
        self.assertIn("afterok", submit)
        self.assertIn("--array=0-9%2", submit)
        self.assertIn("precheck_nf_generalize_fig2_dit_l16_fresh300k.sbatch", submit)
        self.assertIn("dpm50_fresh_${total_k}k", submit)
        self.assertIn("START_STAGE=${START_STAGE:-1}", submit)
        self.assertIn("REUSE_EXISTING_MANIFEST=${REUSE_EXISTING_MANIFEST:-0}", submit)

    def test_result_audit_rejects_duplicate_metric_tags(self):
        module = load_result_audit_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            path.write_text(
                "dataset_tag,gen_gl_q95\n"
                "d2p06,0.1\n"
                "d2p06,0.2\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.audit_metrics(path, {"d2p06"})

    def test_result_audit_checks_sample_contract_and_writes_summary(self):
        source = RESULT_AUDIT_PATH.read_text()
        self.assertIn("DPMSolverMultistepScheduler", source)
        self.assertIn("expected_sample_shape", source)
        self.assertIn("np.isfinite", source)
        self.assertIn("summary_output", source)


if __name__ == "__main__":
    unittest.main()
