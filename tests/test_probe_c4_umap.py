from __future__ import annotations

import ast
import importlib.util
import pickle
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_measured_power_deficit_depth_uses_saved_finite_ratio_minimum():
    module = load_module()

    assert module.measured_power_deficit_depth([1.02, 0.65, np.nan]) == pytest.approx(0.35)
    assert module.measured_power_deficit_depth([1.02, 1.01]) == 0.0
    with pytest.raises(ValueError, match="finite"):
        module.measured_power_deficit_depth([np.nan, np.inf])


def test_identity_requires_array_equality_and_both_zero_width_intervals():
    module = load_module()
    original = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    same = original + np.float32(5e-8)
    metric = {
        "centroid_distance_ci_low": 0.0,
        "centroid_distance_ci_high": 0.0,
        "knn_cross_source_fraction_ci_low": 0.5,
        "knn_cross_source_fraction_ci_high": 0.5,
    }

    identity = module.classify_transform_identity(original, same, metric)
    assert identity == {
        "transform_arrays_allclose": True,
        "centroid_ci_zero_width": True,
        "knn_ci_zero_width": True,
        "transform_is_identity": True,
        "identity_reason": "transform had no effect at this N",
    }

    shifted = module.classify_transform_identity(original, original + 0.1, metric)
    assert shifted["transform_arrays_allclose"] is False
    assert shifted["transform_is_identity"] is False
    assert "arrays differ" in shifted["identity_reason"]

    uncertain = module.classify_transform_identity(
        original,
        same,
        {**metric, "knn_cross_source_fraction_ci_high": 0.51},
    )
    assert uncertain["knn_ci_zero_width"] is False
    assert uncertain["transform_is_identity"] is False

    assert module.generated_identity_diagnostics() == {
        "transform_arrays_allclose": False,
        "centroid_ci_zero_width": False,
        "knn_ci_zero_width": False,
        "transform_is_identity": False,
        "identity_reason": "not applicable: generated samples are not a transform arm",
    }


def test_perfect_mixing_expectation_is_exact_for_finite_populations():
    module = load_module()

    assert module.perfect_mixing_expectation(4, 4) == pytest.approx(4 / 7)
    assert module.perfect_mixing_expectation(3, 5) == pytest.approx(30 / 56)
    with pytest.raises(ValueError, match="positive"):
        module.perfect_mixing_expectation(0, 4)


def test_real_split_baseline_is_balanced_deterministic_and_bootstrapped():
    module = load_module()
    simulations = np.repeat(np.arange(900, 904), 4)
    features = np.column_stack(
        [simulations.astype(float), np.tile(np.arange(4, dtype=float), 4)]
    )

    left, right, membership = module.deterministic_balanced_real_split(
        simulations, seed=123
    )
    left_again, right_again, membership_again = module.deterministic_balanced_real_split(
        simulations, seed=123
    )
    np.testing.assert_array_equal(left, left_again)
    np.testing.assert_array_equal(right, right_again)
    assert membership == membership_again
    assert membership["seed"] == 123
    assert membership["rule"] == "seeded within-simulation balanced half split"
    for sim in np.unique(simulations):
        assert np.sum(simulations[left] == sim) == 2
        assert np.sum(simulations[right] == sim) == 2

    baseline = module.real_split_mixing_baseline(
        features,
        simulations,
        k=3,
        n_boot=100,
        seed=123,
    )
    assert 0.0 <= baseline["real_split_mixing_baseline"] <= 1.0
    assert baseline["real_split_mixing_baseline_ci_low"] <= baseline[
        "real_split_mixing_baseline"
    ]
    assert baseline["real_split_mixing_baseline"] <= baseline[
        "real_split_mixing_baseline_ci_high"
    ]
    assert baseline["split_membership"] == membership

    with pytest.raises(ValueError, match="even"):
        module.deterministic_balanced_real_split(
            np.array([900, 900, 900, 901, 901, 901]), seed=123
        )


def test_c4_v3_tables_separate_complete_headline_identity_and_visual_only():
    script = load_script_module()
    simulations = np.repeat(np.arange(900, 904), 2)
    reference = np.column_stack(
        [np.repeat(np.arange(4, dtype=float), 2), np.tile([0.0, 0.1], 4)]
    )
    run_name = "nf_cond_bias_hi_u128_d2p07_n128_200k"
    sources = (
        ("shared_real_original", "real_original", reference),
        (run_name, "real_measured_transfer", reference.copy()),
        (run_name, "real_gaussian", reference + np.array([0.0, 4.0])),
        (run_name, "generated", reference + np.array([0.0, 8.0])),
    )
    metadata = pd.concat(
        [
            pd.DataFrame(
                {
                    "run_name": [row_run] * len(values),
                    "source": [source] * len(values),
                    "sim_index": simulations,
                    "transform": [f"{source}__transform"] * len(values),
                }
            )
            for row_run, source, values in sources
        ],
        ignore_index=True,
    )
    standardized = np.concatenate([values for _, _, values in sources])
    original_images = np.zeros((len(reference), 1, 2, 2), dtype=np.float32)
    transformed = {
        (run_name, "real_measured_transfer"): original_images.copy(),
        (run_name, "real_gaussian"): np.ones_like(original_images),
    }
    run_parameters = {
        run_name: {
            "dataset_size": 128,
            "gaussian_sigma_pixels": 0.75,
            "measured_power_deficit_depth": 0.35,
        }
    }

    complete, headline, identities, baseline = script.build_c4_metric_tables(
        metadata=metadata,
        standardized=standardized,
        original_images=original_images,
        transformed_images=transformed,
        run_parameters=run_parameters,
        k=1,
        n_boot=100,
        seed=123,
    )

    assert len(complete) == 3
    assert set(complete["feature_space"]) == {
        "frozen_standardized_vgg_mlp_input_1024d"
    }
    assert set(complete["gaussian_sigma_pixels"]) == {0.75}
    assert set(complete["measured_power_deficit_depth"]) == {0.35}
    assert set(complete["source_count"]) == {8}
    assert set(complete["reference_count"]) == {8}
    np.testing.assert_allclose(complete["perfect_mixing_expectation"], 8 / 15)
    assert len(headline) == 2
    assert "real_measured_transfer" not in set(headline["source"])
    measured = complete.set_index("source").loc["real_measured_transfer"]
    assert bool(measured["transform_is_identity"]) is True
    assert measured["identity_reason"] == "transform had no effect at this N"
    assert len(identities) == 2
    assert identities.set_index("source").loc[
        "real_measured_transfer", "message"
    ] == "transform had no effect at this N"
    assert baseline["split_membership"]["seed"] == 123

    embedding = standardized[:, :2]
    visual = script.build_umap_visual_only_table(metadata, embedding)
    assert "umap_layout_separation_visual_only" in visual.columns
    assert not any("centroid" in column for column in visual.columns)


def test_umap_connectivity_warning_is_recorded_and_other_warnings_are_visible():
    script = load_script_module()

    class FakeReducer:
        def fit_transform(self, values):
            warnings.warn("Graph is not fully connected; fixture", UserWarning)
            warnings.warn("unexpected reducer warning", RuntimeWarning)
            return np.zeros((len(values), 2), dtype=np.float32)

    with pytest.warns(RuntimeWarning, match="unexpected reducer warning"):
        embedding, report = script.fit_umap_with_connectivity_report(
            FakeReducer(), np.ones((4, 3), dtype=np.float32)
        )

    assert embedding.shape == (4, 2)
    assert report == {
        "umap_graph_not_fully_connected": True,
        "warnings": [
            {
                "category": "UserWarning",
                "message": "Graph is not fully connected; fixture",
            }
        ],
    }

    class QuietReducer:
        def fit_transform(self, values):
            return np.zeros((len(values), 2), dtype=np.float32)

    _, quiet = script.fit_umap_with_connectivity_report(
        QuietReducer(), np.ones((3, 2), dtype=np.float32)
    )
    assert quiet == {
        "umap_graph_not_fully_connected": False,
        "warnings": [],
    }


def test_threadpool_compatibility_is_narrow_and_typedstorage_filter_is_scoped():
    script = load_script_module()

    def known_failure():
        raise AttributeError("'NoneType' object has no attribute 'split'")

    report = script.threadpool_compatibility_report(known_failure)
    assert report["status"] == "known_callback_failure"
    assert report["exception_type"] == "AttributeError"

    def unrelated_failure():
        raise AttributeError("unrelated bug")

    with pytest.raises(AttributeError, match="unrelated bug"):
        script.threadpool_compatibility_report(unrelated_failure)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        script.configure_known_warning_filters()
        warnings.warn("TypedStorage is deprecated and will be removed", UserWarning)
        warnings.warn("another PyTorch warning", UserWarning)
    assert [str(item.message) for item in caught] == ["another PyTorch warning"]


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


def test_corrected_c4_run_records_the_v3_analysis_identity():
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
    assert config["analysis"] == "c4_frozen_vgg_umap_seed123_v3"


def test_long_umap_job_has_frozen_code_and_new_output_guards():
    source = SBATCH_PATH.read_text()
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --time=02:00:00" in source
    assert "EXPECTED_COMMIT" in source
    assert 'git -C "${CODE_ROOT}" status --porcelain' in source
    assert "c4_frozen_vgg_umap_seed123_v3" in source
    assert 'test ! -e "${OUTPUT_DIR}"' in source
    assert "finalize_terminal_report.py" in source
    assert "--status FAILED" in source
    assert "--status PASS" in source
    assert "c4_feature_metrics_complete.csv" in source
    assert "c4_feature_metrics_headline.csv" in source
    assert "c4_transform_identity_report.csv" in source
    assert "umap_layout_separation_visual_only.csv" in source
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


def test_mixing_baseline_plot_is_written_from_complete_metrics(tmp_path):
    script = load_script_module()
    table = pd.DataFrame(
        {
            "run_name": ["run-a", "run-a"],
            "dataset_size": [128, 128],
            "source": ["real_measured_transfer", "generated"],
            "knn_cross_source_fraction": [0.41, 0.22],
            "knn_cross_source_fraction_ci_low": [0.38, 0.18],
            "knn_cross_source_fraction_ci_high": [0.44, 0.27],
            "perfect_mixing_expectation": [0.5001, 0.5001],
            "real_split_mixing_baseline": [0.49, 0.49],
            "transform_is_identity": [True, False],
        }
    )
    output = tmp_path / "mixing.png"

    script.plot_mixing_baselines(table, output)

    assert output.is_file()
    assert output.stat().st_size > 1_000


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
