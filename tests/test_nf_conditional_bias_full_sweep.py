from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _module(name: str):
    return importlib.import_module(name)


def _synthetic_tables():
    prep = _module("prepare_nf_conditional_bias_full_sweep_configs")
    parameters = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
    point_rows = []
    slope_rows = []
    for dataset_size in prep.ALL_DATASET_SIZES:
        exponent = int(np.log2(dataset_size))
        slope = 0.15 + 0.08 * (exponent - 6)
        for parameter in parameters:
            for theta_in in np.linspace(0.15, 0.50, 8):
                recovered = 0.20 + slope * (theta_in - 0.20)
                point_rows.append(
                    {
                        "run_name": prep.run_name(dataset_size),
                        "dataset_size": dataset_size,
                        "parameter": parameter,
                        "theta_in": theta_in,
                        "theta_rec_median": recovered,
                        "theta_rec_q16": recovered - 0.02,
                        "theta_rec_q84": recovered + 0.02,
                    }
                )
            slope_rows.append(
                {
                    "run_name": prep.run_name(dataset_size),
                    "dataset_size": dataset_size,
                    "parameter": parameter,
                    "slope": slope,
                    "slope_ci16": slope - 0.04,
                    "slope_ci84": slope + 0.04,
                    "intercept": 0.20 * (1.0 - slope),
                }
            )
    return pd.DataFrame(point_rows), pd.DataFrame(slope_rows)


def test_full_sweep_trains_all_ten_sizes_from_clean_initializations():
    prep = _module("prepare_nf_conditional_bias_full_sweep_configs")
    assert prep.ALL_DATASET_SIZES == tuple(2**p for p in range(6, 16))
    assert prep.SWEEP_NAME == "nf_conditional_bias_fresh_full_sweep_200k"
    assert prep.REUSED_DATASET_SIZES == ()
    assert prep.TRAIN_DATASET_SIZES == prep.ALL_DATASET_SIZES
    assert prep.TRAINING_SEED == 123
    assert len({prep.run_name(n) for n in prep.ALL_DATASET_SIZES}) == 10
    assert all("fresh200k" in prep.run_name(n) for n in prep.ALL_DATASET_SIZES)
    assert prep.checkpoint_epoch_for(128) == prep.base.epochs_for(128, prep.base.TARGET_UPDATES) - 1


def test_training_sets_are_deterministic_nested_prefixes():
    prep = _module("prepare_nf_conditional_bias_full_sweep_configs")
    allowed = np.arange(100, dtype=np.int64)
    sizes = (64, 128, 256)
    selected = prep.nested_slice_pairs_by_size(allowed, sizes, z_size=4)
    repeated = prep.nested_slice_pairs_by_size(allowed, sizes, z_size=4)

    for size in sizes:
        assert np.array_equal(selected[size], repeated[size])
        assert len({tuple(pair) for pair in selected[size]}) == size
    for smaller, larger in zip(sizes, sizes[1:]):
        assert set(map(tuple, selected[smaller])).issubset(set(map(tuple, selected[larger])))


def test_sampler_forwards_manifest_pinned_checkpoint_epoch():
    sampler = _module("sample_nf_conditional_bias_probe")
    assert sampler.checkpoint_cli_args({"checkpoint_epoch": 6249}) == [
        "--checkpoint_epoch",
        "6249",
    ]
    assert sampler.checkpoint_cli_args({}) == []


def test_sample_annotation_records_pinned_checkpoint(tmp_path):
    sampler = _module("sample_nf_conditional_bias_probe")
    sample_path = tmp_path / "sample.npz"
    heldout_norm = tmp_path / "heldout_norm.npy"
    heldout_raw = tmp_path / "heldout_raw.npy"
    heldout_indices = tmp_path / "heldout_indices.txt"
    np.savez(sample_path, samples=np.zeros((2, 1, 4, 4), dtype=np.float32))
    np.save(heldout_norm, np.zeros((2, 6), dtype=np.float32))
    np.save(heldout_raw, np.zeros((2, 6), dtype=np.float32))
    np.savetxt(heldout_indices, np.array([900, 901]), fmt="%d")
    row = {
        "run_name": "pinned-run",
        "regime": "full_sweep",
        "dataset_size": 128,
        "checkpoint_epoch": 49_999,
        "requested_checkpoint": "/scratch/checkpoints/checkpoint-epoch-49999",
        "heldout_sample_params_norm_path": str(heldout_norm),
        "heldout_indices_path": str(heldout_indices),
        "heldout_raw_params_path": str(heldout_raw),
    }
    sampler.annotate_npz(sample_path, row, 123, 1, tmp_path, None)
    with np.load(sample_path) as data:
        assert int(data["checkpoint_epoch"]) == 49_999
        assert str(data["requested_checkpoint"]) == row["requested_checkpoint"]


def test_scientific_signature_catches_conditioning_or_preprocessing_changes(tmp_path):
    prep = _module("prepare_nf_conditional_bias_full_sweep_configs")
    base = _module("prepare_nf_conditional_bias_probe_configs")
    config = base.build_config(
        checkpoint_root=tmp_path / "checkpoints",
        prepared_data_root=tmp_path / "prepared",
        dataset_size=128,
        norm_info={"center": 1.2, "xmax": 3.4, "norm_fit_slices": 4096},
        image_file=tmp_path / "images.npy",
        label_file=tmp_path / "labels.npy",
        heldout_label_file=tmp_path / "heldout.npy",
        heldout_count=32,
        sample_k=64,
    )
    assert config["model"]["kwargs"]["encoder_hid_dim"] == 6
    assert prep.scientific_signature(config) == prep.scientific_signature(dict(config))

    changed = json.loads(json.dumps(config))
    changed["data"]["transform"] = []
    with pytest.raises(ValueError, match="scientific protocol"):
        prep.assert_matching_scientific_protocol(config, changed)

    changed = json.loads(json.dumps(config))
    changed["model"]["kwargs"]["encoder_hid_dim"] = 1
    with pytest.raises(ValueError, match="scientific protocol"):
        prep.assert_matching_scientific_protocol(config, changed)


def test_calibration_plots_require_and_show_all_ten_sizes(tmp_path):
    plotting = _module("plot_nf_conditional_bias_full_sweep")
    points, slopes = _synthetic_tables()
    plotting.validate_complete_sizes(points)
    grid = plotting.plot_omega_m_grid(points, slopes, tmp_path / "omega_grid.png")
    transition = plotting.plot_omega_m_transition(
        points,
        slopes,
        tmp_path / "omega_transition.png",
    )
    summary = plotting.plot_parameter_slope_summary(slopes, tmp_path / "slope_summary.png")
    assert grid.exists() and grid.stat().st_size > 10_000
    assert transition.exists() and transition.stat().st_size > 10_000
    assert summary.exists() and summary.stat().st_size > 10_000

    incomplete = points[points["dataset_size"] != 2**10]
    with pytest.raises(ValueError, match="missing dataset sizes"):
        plotting.validate_complete_sizes(incomplete)


def test_slurm_pipeline_is_bounded_and_ordered():
    submit = (ROOT / "scripts/slurm/submit_nf_conditional_bias_full_sweep.sh").read_text()
    precheck = (ROOT / "scripts/slurm/precheck_nf_conditional_bias_full_sweep.sbatch").read_text()
    train = (ROOT / "scripts/slurm/train_nf_conditional_bias_full_sweep_array.sbatch").read_text()
    sample = (ROOT / "scripts/slurm/sample_nf_conditional_bias_full_sweep_array.sbatch").read_text()
    evaluate = (ROOT / "scripts/slurm/evaluate_nf_conditional_bias_full_sweep.sbatch").read_text()
    assert submit.count("--array=0-9%2") == 2
    assert "--time=48:00:00" in train
    assert "--time=08:00:00" in sample
    assert "afterok:${precheck}" in submit
    assert "afterok:${train}" in submit
    assert "afterok:${sample}" in submit
    assert "--encoder-type vgg" in evaluate
    assert "vgg_mlp_encoder.npz" in evaluate
    assert "--checkpoint_epoch" in precheck
    assert "nf_conditional_bias_fresh_full_sweep_200k" in submit
    assert "COSMODIFF_TRAIN_SEED" in train
    assert "torch.manual_seed" in train
    assert "train all ten fresh sizes" in submit


def test_results_notebook_contains_full_sweep_section():
    notebook = json.loads((ROOT / "notebooks/nf_conditional_bias_vgg_results.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )
    assert "Full training-size sweep" in source
    assert "bias_probe_omega_m_all_dataset_sizes.png" in source
    assert "bias_probe_omega_m_transition_vs_dataset_size.png" in source
    assert "bias_probe_all_parameter_slopes_vs_dataset_size.png" in source
    assert "missing dataset sizes" in source
    assert "all ten generators are trained from clean initializations" in source
    assert "checkpoints are reused" not in source
    assert "PROJECT_DIR = Path.cwd().resolve()" in source
    assert "while PROJECT_DIR != PROJECT_DIR.parent" in source


def test_full_sweep_cell_does_not_depend_on_an_earlier_root_variable():
    notebook = json.loads((ROOT / "notebooks/nf_conditional_bias_vgg_results.ipynb").read_text())
    sweep_cells = [
        cell
        for cell in notebook.get("cells", [])
        if "conditional-full-sweep" in cell.get("metadata", {}).get("tags", [])
        and cell.get("cell_type") == "code"
    ]
    assert len(sweep_cells) == 1
    source = "".join(sweep_cells[0].get("source", []))
    assignment = source.index("PROJECT_DIR = Path.cwd().resolve()")
    first_use = source.index("full_sweep_root = PROJECT_DIR")
    assert assignment < first_use


def test_plot_module_imports_in_notebook_package_mode():
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.plot_nf_conditional_bias_full_sweep"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
