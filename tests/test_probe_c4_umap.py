from __future__ import annotations

import ast
import importlib.util
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "simdiff_eval" / "probe_c4_umap.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_probe_c4_umap.py"
SBATCH_PATH = REPO_ROOT / "scripts" / "slurm" / "probe_c4_frozen_vgg_umap.sbatch"
INSTALL_PATH = REPO_ROOT / "scripts" / "install_probe_c4_umap_runtime.sh"


def load_module():
    spec = importlib.util.spec_from_file_location("probe_c4_umap", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_probe_c4_umap", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_mlp_input_uses_fitted_scaler_without_refitting():
    module = load_module()

    class FrozenScaler:
        n_features_in_ = 3

        def __init__(self):
            self.transform_calls = 0
            self.fit_calls = 0

        def transform(self, values):
            self.transform_calls += 1
            return np.asarray(values) + 10.0

        def fit(self, values):
            self.fit_calls += 1
            raise AssertionError("the frozen scaler must never be fitted")

    class FrozenHead:
        def __init__(self, scaler):
            self.named_steps = {"standardscaler": scaler, "mlpregressor": object()}

    scaler = FrozenScaler()
    raw = np.arange(12, dtype=np.float32).reshape(4, 3)
    standardized, report = module.frozen_mlp_inputs(FrozenHead(scaler), raw)

    np.testing.assert_array_equal(standardized, raw + 10.0)
    assert scaler.transform_calls == 1
    assert scaler.fit_calls == 0
    assert report["feature_dim"] == 3
    assert report["scaler_class"] == "FrozenScaler"

    class WrongOrderHead:
        named_steps = {"mlpregressor": object(), "standardscaler": scaler}

    try:
        module.frozen_mlp_inputs(WrongOrderHead(), raw)
    except TypeError as exc:
        assert "order" in str(exc)
    else:
        raise AssertionError("a head with reversed pipeline steps was accepted")


def test_balanced_real_pairs_are_fixed_even_slices_for_all_heldout_sims():
    module = load_module()
    heldout = np.arange(900, 932, dtype=np.int64)
    pairs = module.balanced_real_slice_pairs(heldout, slices_per_sim=64)

    assert pairs.shape == (32 * 64, 2)
    for sim in heldout:
        selected = pairs[pairs[:, 0] == sim, 1]
        np.testing.assert_array_equal(selected, np.arange(0, 128, 2))


def test_centroid_and_knn_metrics_use_balanced_simulation_blocks():
    module = load_module()
    sims = np.repeat(np.arange(900, 904), 2)
    reference = np.column_stack([np.repeat(np.arange(4), 2), np.zeros(8)])
    same = reference.copy()
    far = reference + np.array([0.0, 20.0])

    same_result = module.compare_source_to_reference(
        same,
        reference,
        source_sim=sims,
        reference_sim=sims,
        k=3,
        n_boot=200,
        seed=123,
    )
    far_result = module.compare_source_to_reference(
        far,
        reference,
        source_sim=sims,
        reference_sim=sims,
        k=3,
        n_boot=200,
        seed=123,
    )

    assert same_result["centroid_distance"] == 0.0
    assert same_result["knn_cross_source_fraction"] > 0.3
    assert far_result["centroid_distance"] == 20.0
    assert far_result["knn_cross_source_fraction"] == 0.0
    for key in (
        "centroid_distance_ci_low",
        "centroid_distance_ci_high",
        "knn_cross_source_fraction_ci_low",
        "knn_cross_source_fraction_ci_high",
    ):
        assert np.isfinite(far_result[key])


def test_knn_mixing_keeps_the_actual_nearest_nonself_neighbour():
    module = load_module()
    result = module.compare_source_to_reference(
        np.array([[0.0], [10.0]]),
        np.array([[1.0], [11.0]]),
        source_sim=np.array([900, 901]),
        reference_sim=np.array([900, 901]),
        k=1,
        n_boot=20,
        seed=123,
    )
    assert result["knn_cross_source_fraction"] == 1.0


def test_feature_rows_preserve_required_sample_provenance():
    module = load_module()
    metadata = module.source_metadata(
        run_name="nf_cond_bias_hi_u128_d2p07_n128_200k",
        dataset_size=128,
        source="real_measured_transfer",
        transform="transfer_Tk__run__N128",
        sim_index=np.array([900, 901]),
        sample_index=np.array([0, 0]),
        slice_index=np.array([4, 6]),
        omega_m=np.array([0.2, 0.3]),
        code_revision="abc123",
        config_sha256="def456",
    )

    assert list(metadata.columns) == module.PROVENANCE_COLUMNS
    assert metadata.to_dict("records")[0] == {
        "run_name": "nf_cond_bias_hi_u128_d2p07_n128_200k",
        "dataset_size": 128,
        "sim_index": 900,
        "sample_index": 0,
        "slice_index": 4,
        "transform": "transfer_Tk__run__N128",
        "source": "real_measured_transfer",
        "Omega_m": 0.2,
        "code_revision": "abc123",
        "config_sha256": "def456",
    }


def test_analysis_applies_saved_c4_parameters_and_never_rederives_them():
    source = SCRIPT_PATH.read_text()
    assert "saved[\"measured_transfer\"]" in source
    assert "saved[\"gaussian_sigma_pixels\"]" in source
    assert "power_ratio_transfer" not in source
    assert "fit_gaussian_smoothing" not in source
    assert "reducer.fit_transform(standardized)" in source
    assert "scaler.transform(features)" in MODULE_PATH.read_text()
    assert "scaler.fit(" not in MODULE_PATH.read_text()


def test_main_uses_the_requested_device_before_the_load_report_exists():
    tree = ast.parse(SCRIPT_PATH.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_frozen_vgg_and_head"
    ]
    assert len(calls) == 1
    device = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "device")
    assert isinstance(device, ast.Attribute)
    assert isinstance(device.value, ast.Name)
    assert (device.value.id, device.attr) == ("args", "device")


def test_repaired_c4_run_records_the_v2_analysis_identity():
    script = load_script_module()
    config = script._analysis_config(
        types.SimpleNamespace(
            umap_neighbors=30,
            umap_min_dist=0.1,
            seed=123,
            knn_k=15,
            bootstrap=2000,
        ),
        [{"run_name": "run-a"}],
    )
    assert config["analysis"] == "c4_frozen_vgg_umap_seed123_v2"


def test_long_umap_job_has_frozen_code_and_new_output_guards():
    source = SBATCH_PATH.read_text()
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --time=02:00:00" in source
    assert "EXPECTED_COMMIT" in source
    assert 'git -C "${CODE_ROOT}" status --porcelain' in source
    assert "c4_frozen_vgg_umap_seed123_v2" in source
    assert 'test ! -e "${OUTPUT_DIR}"' in source
    assert "--expected-results-revision" in source
    assert "UMAP_SITE_PACKAGES" in source
    assert 'umap.__version__ == "0.5.5"' in source
    assert 'pynndescent.__version__ == "0.5.10"' in source
    assert 'numba.__version__ == "0.59.1"' in source
    assert 'llvmlite.__version__ == "0.42.0"' in source
    assert "Path(umap.__file__).resolve().is_relative_to(runtime)" in source
    assert "Path(pynndescent.__file__).resolve().is_relative_to(runtime)" in source
    assert "Path(numba.__file__).resolve().is_relative_to(runtime)" in source
    assert "Path(llvmlite.__file__).resolve().is_relative_to(runtime)" in source


def test_umap_runtime_installer_is_pinned_isolated_and_does_not_modify_the_venv():
    source = INSTALL_PATH.read_text()
    assert "umap-learn==0.5.5" in source
    assert "pynndescent==0.5.10" in source
    assert "numba==0.59.1" in source
    assert "llvmlite==0.42.0" in source
    assert "--no-deps" in source
    assert "--target" in source
    assert "mktemp -d" in source
    assert "UMAP_SITE_PACKAGES" in source
    assert 'umap.__version__ == "0.5.5"' in source
    assert 'pynndescent.__version__ == "0.5.10"' in source
    assert 'numba.__version__ == "0.59.1"' in source
    assert 'llvmlite.__version__ == "0.42.0"' in source
    assert 'numpy.__version__ == "1.26.4"' in source
    assert '--target "${TEMP_DIR}"' in source
    assert "--upgrade" not in source


def test_explicit_head_and_vgg_weight_files_are_the_objects_actually_loaded(
    tmp_path, monkeypatch
):
    script = load_script_module()
    weights_path = tmp_path / "vgg16-397923af.pth"
    torch.save({"exact": torch.tensor([7.0])}, weights_path)
    head_path = tmp_path / "exact_head.pkl"
    with head_path.open("wb") as handle:
        pickle.dump({"identity": "exact frozen head"}, handle)

    class FakeModel:
        def __init__(self):
            self.features = object()
            self.loaded = None

        def load_state_dict(self, state, strict):
            self.loaded = (state, strict)

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

        def parameters(self):
            return []

    fake_model = FakeModel()
    fake_models = types.ModuleType("torchvision.models")
    fake_models.vgg16 = lambda weights=None: fake_model
    monkeypatch.setitem(sys.modules, "torchvision.models", fake_models)

    vgg, head, report = script.load_frozen_vgg_and_head(
        weights_path=weights_path,
        head_path=head_path,
        device="cpu",
    )

    assert vgg is fake_model.features
    assert head == {"identity": "exact frozen head"}
    assert fake_model.loaded[1] is True
    assert float(fake_model.loaded[0]["exact"].item()) == 7.0
    assert report["weights_path"] == str(weights_path.resolve())
    assert report["head_path"] == str(head_path.resolve())


def test_generated_sample_path_is_pinned_to_results_root(tmp_path):
    script = load_script_module()
    results_root = tmp_path / "nfs_results"
    row = {
        "sample_path": (
            "results/nf_conditional_bias_probe/samples/"
            "run_seed{seed}_dpm50_heldout_k{k}.npz"
        )
    }

    resolved = script.generated_sample_path(
        results_root,
        row,
        seed=123,
        samples_per_cosmology=64,
    )

    assert resolved == results_root / "samples/run_seed123_dpm50_heldout_k64.npz"


def test_explicit_probe_artifacts_must_match_completed_c4_manifest():
    script = load_script_module()
    script.validate_c4_probe_artifacts(
        {"encoder": {"sha256": "enc"}, "head": {"sha256": "head"}},
        encoder_record={"sha256": "enc"},
        head_record={"sha256": "head"},
    )
    try:
        script.validate_c4_probe_artifacts(
            {"encoder": {"sha256": "enc"}, "head": {"sha256": "old"}},
            encoder_record={"sha256": "enc"},
            head_record={"sha256": "new"},
        )
    except RuntimeError as exc:
        assert "head" in str(exc)
    else:
        raise AssertionError("mismatched frozen head was accepted")


def test_power_file_parameters_must_match_saved_c4_manifest_records():
    script = load_script_module()
    run_name = "run_a"
    rows = [{"run_name": run_name, "dataset_size": 128}]
    power = {
        "runs": {
            run_name: {
                "dataset_size": 128,
                "k_bins": [1.0, 2.0],
                "measured_transfer": [0.8, 0.7],
                "gaussian_sigma_pixels": 1.25,
            }
        }
    }
    manifest = {
        "transforms": [
            {
                "name": "transfer_Tk__run_a__N128",
                "family": "transfer",
                "k_bins": [1.0, 2.0],
                "transfer_values": [0.8, 0.7],
            },
            {
                "name": "gaussian_smoothing__run_a__N128",
                "family": "gaussian",
                "sigma_pixels": 1.25,
            },
        ]
    }
    script.validate_saved_c4_parameters(power, manifest, rows)

    power["runs"][run_name]["measured_transfer"][1] = 0.1
    try:
        script.validate_saved_c4_parameters(power, manifest, rows)
    except RuntimeError as exc:
        assert "measured transfer" in str(exc)
    else:
        raise AssertionError("stale C4 power parameters were accepted")
