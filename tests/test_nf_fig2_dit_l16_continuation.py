import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "prepare_nf_generalize_fig2_dit_l16_continue_configs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dit_l16_continue_for_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DitL16ContinuationTests(unittest.TestCase):
    def test_four_stages_cover_five_small_data_runs(self):
        module = load_module()
        args = Namespace(
            checkpoint_root=None,
            dataset_tag=None,
            run_name=None,
            stage_updates=25_000,
            stages=4,
            safety_checkpoint_updates=5_000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_root = Path(tmpdir)
            args.checkpoint_root = checkpoint_root
            for row in module.base.iter_runs():
                if row["arch"] == "dit_l16" and row["dataset_tag"] in module.DEFAULT_DATASET_TAGS:
                    path = checkpoint_root / f'{row["run_name"]}_checkpoints' / "checkpoint-epoch-0099"
                    path.mkdir(parents=True)

            rows = module.continue_rows(args)

        self.assertEqual(len(rows), 20)
        self.assertEqual(sorted({row["continue_stage"] for row in rows}), [1, 2, 3, 4])
        self.assertEqual(sorted({row["dataset_tag"] for row in rows}), list(module.DEFAULT_DATASET_TAGS))
        for row in rows:
            self.assertEqual(row["arch"], "dit_l16")
            self.assertEqual(row["cumulative_target_updates"], row["continue_stage"] * 25_000)
            self.assertGreater(row["final_num_epochs"], row["latest_checkpoint_epoch_at_prepare"])
            self.assertLessEqual(row["checkpoint_every_actual_updates"], 5_000 + row["steps_per_epoch"])

    def test_continue_config_keeps_model_and_training_invariants(self):
        module = load_module()
        base_row = next(
            row for row in module.base.iter_runs()
            if row["arch"] == "dit_l16" and row["dataset_tag"] == "d2p08"
        )
        row = dict(base_row)
        row.update(
            final_num_epochs=1234,
            checkpoint_every_n_epochs=17,
        )

        config = module.build_continue_config(row)

        self.assertEqual(config["model"]["kwargs"]["num_layers"], 16)
        self.assertEqual(config["noise_scheduler"]["kwargs"]["prediction_type"], "v_prediction")
        self.assertEqual(config["data"]["constant_label"], 0)
        self.assertEqual(config["train"]["gradient_accumulation_steps"], 4)
        self.assertEqual(config["train"]["num_epochs"], 1234)
        self.assertEqual(config["train"]["checkpoint_every_n_epochs"], 17)

    def test_slurm_workflow_is_staged_and_samples_exact_checkpoint(self):
        train = (REPO_ROOT / "scripts" / "slurm" / "train_nf_generalize_fig2_dit_l16_continue_array.sbatch").read_text()
        sample = (REPO_ROOT / "scripts" / "slurm" / "sample_nf_generalize_fig2_dit_l16_continue_array.sbatch").read_text()
        submit = (REPO_ROOT / "scripts" / "slurm" / "submit_nf_generalize_fig2_dit_l16_continue.sh").read_text()

        self.assertIn("#SBATCH --time=08:00:00", train)
        self.assertIn("CONTINUE_STAGE=${CONTINUE_STAGE:?", train)
        self.assertIn("expected_checkpoint", sample)
        self.assertIn('--checkpoint "${CHECKPOINT_PATH}"', sample)
        self.assertIn("afterok", submit)
        self.assertIn("for stage in 1 2 3 4", submit)
        self.assertIn("--array=0-4%2", submit)

    def test_analyzers_accept_the_frozen_small_data_manifest(self):
        pca = (REPO_ROOT / "scripts" / "slurm" / "analyze_nf_generalize_fig2_dit_pca.sbatch").read_text()
        sscd = (REPO_ROOT / "scripts" / "slurm" / "analyze_nf_generalize_fig2_dit_sscd.sbatch").read_text()
        submit = (REPO_ROOT / "scripts" / "slurm" / "submit_nf_generalize_fig2_dit_l16_continue.sh").read_text()

        self.assertIn("MANIFEST_PATH=${MANIFEST_PATH:-", pca)
        self.assertIn('--manifest "${MANIFEST_PATH}"', pca)
        self.assertIn("MANIFEST_PATH=${MANIFEST_PATH:-", sscd)
        self.assertIn('--manifest "${MANIFEST_PATH}"', sscd)
        self.assertIn("analysis_manifest.json", submit)
        self.assertIn('SAMPLE_LABEL="dpm50"', submit)

    def test_results_notebook_tracks_all_continuation_checkpoints(self):
        notebook = json.loads(
            (REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb").read_text()
        )
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("CONTINUE_MANIFEST_PATH", source)
        for label in ("dpm50_cont_225k", "dpm50_cont_250k", "dpm50_cont_275k", "dpm50_cont_300k"):
            self.assertIn(label, source)
        self.assertIn("continuation_fidelity_df", source)
        self.assertIn("max_raw_samples=None", source)


if __name__ == "__main__":
    unittest.main()
