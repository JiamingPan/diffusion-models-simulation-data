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
    def _args(self, checkpoint_root, continuation_checkpoint_root):
        return Namespace(
            checkpoint_root=checkpoint_root,
            continuation_checkpoint_root=continuation_checkpoint_root,
            dataset_tag=None,
            run_name=None,
            stage_updates=25_000,
            stages=4,
            safety_checkpoint_updates=5_000,
        )

    def _create_original_checkpoints(self, module, checkpoint_root, *, add_overshoot=False):
        selected = []
        for row in module.base.iter_runs():
            if row["arch"] != "dit_l16" or row["dataset_tag"] not in module.DEFAULT_DATASET_TAGS:
                continue
            selected.append(row)
            checkpoint_dir = checkpoint_root / f'{row["run_name"]}_checkpoints'
            base_epoch = int(row["epochs"]) - 1
            (checkpoint_dir / f"checkpoint-epoch-{base_epoch:04d}").mkdir(parents=True)
            if add_overshoot:
                (checkpoint_dir / f"checkpoint-epoch-{base_epoch + 10_000:04d}").mkdir()
        return selected

    def test_four_stages_use_exact_base_checkpoint_and_isolated_directories(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "original"
            continuation_root = root / "continuation"
            base_rows = self._create_original_checkpoints(
                module, checkpoint_root, add_overshoot=True
            )
            args = self._args(checkpoint_root, continuation_root)
            rows = module.continue_rows(args)

        self.assertEqual(len(rows), 20)
        self.assertEqual(sorted({row["continue_stage"] for row in rows}), [1, 2, 3, 4])
        self.assertEqual(sorted({row["dataset_tag"] for row in rows}), list(module.DEFAULT_DATASET_TAGS))
        base_by_tag = {row["dataset_tag"]: row for row in base_rows}
        for row in rows:
            base_row = base_by_tag[row["dataset_tag"]]
            base_epoch = int(base_row["epochs"]) - 1
            stage = int(row["continue_stage"])
            steps = int(row["steps_per_epoch"])
            previous_epoch = base_epoch + module.math.ceil((stage - 1) * 25_000 / steps)
            target_epoch = base_epoch + module.math.ceil(stage * 25_000 / steps)

            self.assertEqual(row["arch"], "dit_l16")
            self.assertEqual(row["manifest_version"], 2)
            self.assertEqual(row["cumulative_target_updates"], row["continue_stage"] * 25_000)
            self.assertEqual(row["base_checkpoint_epoch"], base_epoch)
            self.assertEqual(row["previous_expected_epoch"], previous_epoch)
            self.assertEqual(row["expected_final_epoch"], target_epoch)
            self.assertEqual(row["stage_additional_epochs"], target_epoch - previous_epoch)
            self.assertEqual(
                Path(row["checkpoint_dir"]),
                continuation_root / f'{row["run_name"]}_checkpoints',
            )
            self.assertEqual(
                Path(row["base_checkpoint"]),
                checkpoint_root
                / f'{row["run_name"]}_checkpoints'
                / f"checkpoint-epoch-{base_epoch:04d}",
            )
            self.assertEqual(
                Path(row["previous_expected_checkpoint"]).name,
                f"checkpoint-epoch-{previous_epoch:04d}",
            )
            self.assertLessEqual(row["checkpoint_every_actual_updates"], 5_000 + row["steps_per_epoch"])

        d2p07_stage1 = next(
            row for row in rows
            if row["dataset_tag"] == "d2p07" and row["continue_stage"] == 1
        )
        self.assertEqual(d2p07_stage1["base_checkpoint_epoch"], 12_499)
        self.assertEqual(d2p07_stage1["expected_final_epoch"], 14_062)
        self.assertEqual(d2p07_stage1["stage_additional_epochs"], 1_563)

    def test_seed_continuation_directories_links_only_exact_200k_checkpoint(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "original"
            continuation_root = root / "continuation"
            self._create_original_checkpoints(module, checkpoint_root, add_overshoot=True)
            rows = module.continue_rows(self._args(checkpoint_root, continuation_root))

            module.seed_continuation_directories(rows)

            for row in rows:
                seed = Path(row["checkpoint_dir"]) / Path(row["base_checkpoint"]).name
                self.assertTrue(seed.is_symlink())
                self.assertEqual(seed.resolve(), Path(row["base_checkpoint"]).resolve())
                seeded_names = sorted(path.name for path in Path(row["checkpoint_dir"]).iterdir())
                self.assertEqual(seeded_names, [Path(row["base_checkpoint"]).name])

    def test_seed_continuation_directory_rejects_conflicting_checkpoint(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "original"
            continuation_root = root / "continuation"
            self._create_original_checkpoints(module, checkpoint_root)
            rows = module.continue_rows(self._args(checkpoint_root, continuation_root))
            row = next(item for item in rows if item["continue_stage"] == 1)
            conflicting_seed = Path(row["checkpoint_dir"]) / Path(row["base_checkpoint"]).name
            conflicting_seed.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "conflicting checkpoint"):
                module.seed_continuation_directories([row])

    def test_existing_manifest_must_use_current_version(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            path = module._manifest_path(project_dir)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps([{"continue_stage": 1}]))

            with self.assertRaisesRegex(ValueError, "manifest version"):
                module._load_existing_rows(project_dir)

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

        self.assertIn("#SBATCH --time=24:00:00", train)
        self.assertIn("CONTINUE_STAGE=${CONTINUE_STAGE:?", train)
        self.assertIn("expected_checkpoint", sample)
        self.assertIn('--checkpoint "${CHECKPOINT_PATH}"', sample)
        self.assertIn("afterok", submit)
        self.assertIn("for stage in 1 2 3 4", submit)
        self.assertIn("--array=0-4%2", submit)
        self.assertIn("START_STAGE=${START_STAGE:-1}", submit)
        self.assertIn("REUSE_EXISTING_MANIFEST=${REUSE_EXISTING_MANIFEST:-0}", submit)
        self.assertIn('if (( stage < START_STAGE ))', submit)
        self.assertIn("precheck_nf_generalize_fig2_dit_l16_resume.sbatch", submit)

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
        self.assertIn("continuation_table_audit_df", source)
        self.assertIn("At least two analyzed checkpoints are required", source)
        self.assertIn("available_novelty_updates", source)
        self.assertIn("Training set size $N_{2D}$", source)
        self.assertIn("one curve per checkpoint", source)
        self.assertIn("nf_generalize_fig2_dit_l16_continuation_by_data_size.png", source)

    def test_results_notebook_continuation_title_has_valid_mathtext(self):
        notebook = json.loads(
            (REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb").read_text()
        )
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertNotIn("$N_{{2D}}={n_label}$", source)
        self.assertIn("$N_{{2D}}$ = {n_label}", source)


if __name__ == "__main__":
    unittest.main()
