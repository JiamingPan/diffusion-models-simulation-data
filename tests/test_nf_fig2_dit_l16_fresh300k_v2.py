import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = (
    REPO_ROOT
    / "scripts"
    / "prepare_nf_generalize_fig2_dit_l16_fresh300k_v2_configs.py"
)


def load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "fresh_l16_300k_v2_prepare_for_test", PREPARE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FreshDitL16V2ManifestTests(unittest.TestCase):
    def test_ten_fresh_runs_target_300k_once(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = module.fresh_rows(
                Namespace(
                    checkpoint_root=Path(tmpdir) / module.SWEEP_NAME,
                    dataset_tag=None,
                    run_name=None,
                    safety_checkpoint_updates=5_000,
                )
            )

        self.assertEqual(len(rows), 10)
        self.assertEqual(
            sorted(row["dataset_tag"] for row in rows),
            [f"d2p{power:02d}" for power in range(6, 16)],
        )
        self.assertEqual({row["target_total_updates"] for row in rows}, {300_000})
        self.assertEqual({row["training_seed"] for row in rows}, {123})
        self.assertTrue(all(row["fresh_initialization"] for row in rows))
        self.assertTrue(all("fresh300k_v2" in row["run_name"] for row in rows))
        self.assertTrue(
            all(row["checkpoint_every_actual_updates"] <= 5_000 + row["optimizer_steps_per_epoch"]
                for row in rows)
        )

    def test_configs_keep_the_l16_training_recipe(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = module.fresh_rows(
                Namespace(
                    checkpoint_root=Path(tmpdir) / module.SWEEP_NAME,
                    dataset_tag=None,
                    run_name=None,
                    safety_checkpoint_updates=5_000,
                )
            )

        config = module.build_config(rows[0])
        self.assertEqual(config["model"]["kwargs"]["num_layers"], 16)
        self.assertEqual(config["train"]["num_epochs"], rows[0]["final_num_epochs"])
        self.assertEqual(
            config["train"]["checkpoint_every_n_epochs"],
            rows[0]["checkpoint_every_n_epochs"],
        )


class FreshDitL16V2WorkflowSourceTests(unittest.TestCase):
    def test_training_is_one_48_hour_task_per_run_with_two_gpu_cap(self):
        train = (
            REPO_ROOT
            / "scripts"
            / "slurm"
            / "train_nf_generalize_fig2_dit_l16_fresh300k_v2_array.sbatch"
        ).read_text()
        submit = (
            REPO_ROOT
            / "scripts"
            / "slurm"
            / "submit_nf_generalize_fig2_dit_l16_fresh300k_v2.sh"
        ).read_text()

        self.assertIn("#SBATCH --time=48:00:00", train)
        self.assertIn("#SBATCH --array=0-9%2", train)
        self.assertIn("run_cosmodiff_train_fresh_seeded.py", train)
        self.assertIn("run_cosmodiff_train_with_dit_resume.py", train)
        self.assertIn("patch_cosmodiff_checkpoint_state.py", train)
        self.assertIn("--quarantine-incomplete", train)
        self.assertNotIn('if [[ -d "${EXPECTED_CHECKPOINT}" ]]', train)
        self.assertIn("--array=0-9%2", submit)
        self.assertNotIn("for stage in", submit)

    def test_submit_creates_slurm_log_directory_before_first_job(self):
        submit = (
            REPO_ROOT
            / "scripts"
            / "slurm"
            / "submit_nf_generalize_fig2_dit_l16_fresh300k_v2.sh"
        ).read_text()

        mkdir_index = submit.index(
            'mkdir -p "${PROJECT_DIR}/logs/nf_generalize_fig2_dit_l16_fresh300k_v2"'
        )
        first_sbatch_index = submit.index("sbatch")
        self.assertLess(mkdir_index, first_sbatch_index)

    def test_gpu_precheck_trains_saves_loads_and_resumes(self):
        precheck = (
            REPO_ROOT
            / "scripts"
            / "slurm"
            / "precheck_nf_generalize_fig2_dit_l16_fresh300k_v2.sbatch"
        ).read_text()

        self.assertIn("#SBATCH --partition=spgpu", precheck)
        self.assertIn("patch_cosmodiff_checkpoint_state.py", precheck)
        self.assertIn("run_cosmodiff_train_fresh_seeded.py", precheck)
        self.assertIn("check_nf_generalize_fig2_dit_resume.py", precheck)
        self.assertIn("run_cosmodiff_train_with_dit_resume.py", precheck)
        self.assertIn("checkpoint-epoch-0001", precheck)

    def test_sampling_requires_a_strictly_loadable_final_checkpoint(self):
        sample = (
            REPO_ROOT
            / "scripts"
            / "slurm"
            / "sample_nf_generalize_fig2_dit_l16_fresh300k_v2_array.sbatch"
        ).read_text()

        self.assertIn("check_nf_generalize_fig2_dit_resume.py", sample)
        self.assertIn('--checkpoint "${CHECKPOINT_PATH}"', sample)


if __name__ == "__main__":
    unittest.main()
