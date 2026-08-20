from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_load_heldout_real_slices_reproduces_legacy_pair_order(monkeypatch):
    from simdiff_eval import probe_eval
    from train_nf_conditional_bias_encoder import select_slice_pairs

    heldout = np.array([900, 901, 902], dtype=np.int64)
    expected_pairs = select_slice_pairs(heldout, len(heldout) * 7)
    expected_raw = np.arange(len(expected_pairs) * 4, dtype=np.float32).reshape(
        len(expected_pairs), 1, 2, 2
    ) + 1.0
    params = np.arange(1000 * 6, dtype=np.float32).reshape(1000, 6)
    observed: dict[str, np.ndarray] = {}

    monkeypatch.setattr(probe_eval, "image_path", lambda root: Path(root) / "grid.npy")
    monkeypatch.setattr(probe_eval, "params_path", lambda root: Path(root) / "params.txt")
    monkeypatch.setattr(probe_eval, "load_params", lambda path, count: params)

    def fake_load_raw(path, pairs):
        observed["pairs"] = pairs.copy()
        return expected_raw

    monkeypatch.setattr(probe_eval, "load_raw_slices", fake_load_raw)
    monkeypatch.setattr(
        probe_eval,
        "preprocess_real_slices",
        lambda raw, norm: raw.astype(np.float32) * np.float32(norm["scale"]),
    )

    images, theta_raw, sim_index, z_index = probe_eval.load_heldout_real_slices(
        "/synthetic", heldout, 7, {"scale": 0.5}
    )

    np.testing.assert_array_equal(observed["pairs"], expected_pairs)
    np.testing.assert_array_equal(sim_index, expected_pairs[:, 0])
    np.testing.assert_array_equal(z_index, expected_pairs[:, 1])
    np.testing.assert_array_equal(theta_raw, params[expected_pairs[:, 0]])
    np.testing.assert_array_equal(images, expected_raw * np.float32(0.5))
    assert images.dtype == np.float32


class FakeEncoder:
    model_path = Path("fake-head.pkl")

    def predict_norm(self, images, batch_size=512):
        means = np.asarray(images, dtype=np.float32).mean(axis=(1, 2, 3))
        return np.stack([means + offset for offset in range(6)], axis=1)

    def norm_to_raw(self, theta_norm):
        return np.asarray(theta_norm, dtype=np.float32)


def synthetic_probe_inputs():
    images = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 1, 8, 8)
    images = images / images.max()
    sim_index = np.repeat(np.array([900, 901]), 4)
    z_index = np.tile(np.arange(4), 2)
    theta_raw = np.stack(
        [
            np.linspace(0.1 + parameter, 0.2 + parameter, 8, dtype=np.float32)
            for parameter in range(6)
        ],
        axis=1,
    )
    return images, theta_raw, sim_index, z_index


def test_transform_evaluation_has_required_long_columns_and_one_identity():
    from simdiff_eval.probe_controls import TransformSpec, evaluate_transform_specs
    from simdiff_eval.probe_transforms import get_transform

    images, theta_raw, sim_index, z_index = synthetic_probe_inputs()
    specs = [
        TransformSpec("identity", "identity", get_transform("identity")),
        TransformSpec("flip_h", "dihedral", get_transform("flip_h")),
    ]
    table = evaluate_transform_specs(
        images,
        theta_raw,
        sim_index,
        z_index,
        FakeEncoder(),
        specs,
        batch_size=4,
    )
    required = {
        "transform",
        "transform_family",
        "k_cut",
        "k_cut_over_knyq",
        "window",
        "sim_index",
        "z_index",
        "parameter",
        "theta_true",
        "theta_pred",
        "out_of_range_fraction",
    }
    assert required.issubset(table.columns)
    assert table[table["transform"] == "identity"].shape[0] == 8 * 6
    assert table.shape[0] == 2 * 8 * 6


def test_aggregation_reports_per_slice_and_per_cosmology_with_bootstrap():
    from simdiff_eval.probe_controls import (
        TransformSpec,
        aggregate_prediction_table,
        evaluate_transform_specs,
    )
    from simdiff_eval.probe_transforms import get_transform

    images, theta_raw, sim_index, z_index = synthetic_probe_inputs()
    table = evaluate_transform_specs(
        images,
        theta_raw,
        sim_index,
        z_index,
        FakeEncoder(),
        [TransformSpec("identity", "identity", get_transform("identity"))],
        batch_size=4,
    )
    report = aggregate_prediction_table(table, n_boot=50, seed=17)
    grains = {row["grain"] for row in report["metrics"]}
    assert grains == {"per_slice", "per_cosmology"}
    omega_rows = [row for row in report["metrics"] if row["parameter"] == "Omega_m"]
    assert {row["n"] for row in omega_rows} == {2, 8}
    assert all("rmse_ci_low" in row and "slope_ci_high" in row for row in omega_rows)


def test_manifest_records_frozen_encoder_environment_and_seeds(tmp_path, monkeypatch):
    from simdiff_eval import probe_controls

    encoder_path = tmp_path / "encoder.npz"
    head_path = tmp_path / "head.pkl"
    encoder_path.write_bytes(b"encoder artifact")
    head_path.write_bytes(b"head artifact")
    monkeypatch.setattr(
        probe_controls,
        "installed_sklearn_version",
        lambda: "9.9.9",
        raising=False,
    )
    monkeypatch.setattr(
        probe_controls,
        "git_state",
        lambda project_dir: {"revision": "abc123", "dirty": True},
        raising=False,
    )
    manifest = probe_controls.build_run_manifest(
        project_dir=tmp_path,
        encoder_path=encoder_path,
        head_path=head_path,
        heldout_indices=np.arange(900, 932),
        slices_per_sim=128,
        transforms=[{"name": "identity", "family": "identity"}],
        seeds={"bootstrap": 17, "roll": 23},
        arguments={"device": "cpu"},
    )
    assert manifest["git"] == {"revision": "abc123", "dirty": True}
    assert manifest["encoder"]["path"] == str(encoder_path.resolve())
    assert len(manifest["encoder"]["sha256"]) == 64
    assert manifest["head"]["path"] == str(head_path.resolve())
    assert len(manifest["head"]["sha256"]) == 64
    assert manifest["scikit_learn_version"] == "9.9.9"
    assert manifest["heldout_indices"] == list(range(900, 932))
    assert manifest["seeds"] == {"bootstrap": 17, "roll": 23}


def test_json_safe_replaces_nested_nonfinite_values():
    from simdiff_eval.probe_controls import json_safe

    payload = {
        "nan": np.nan,
        "positive_infinity": np.inf,
        "nested": [np.float32(-np.inf), Path("artifact.npz")],
    }
    safe = json_safe(payload)
    encoded = json.dumps(safe, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded == {
        "nan": None,
        "positive_infinity": None,
        "nested": [None, "artifact.npz"],
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "evaluate_probe_transform_controls",
        "evaluate_probe_degradation_control",
    ],
)
def test_report_json_writers_emit_strict_json(module_name, tmp_path):
    module = __import__(module_name)
    output = tmp_path / f"{module_name}.json"
    module._write_json(output, {"finite": 1.0, "missing": np.nan})
    raw = output.read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {"finite": 1.0, "missing": None}


def test_transform_control_cli_help_is_import_safe():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_probe_transform_controls.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "frozen VGG" in result.stdout
    assert "--encoder" in result.stdout
    assert "--control" in result.stdout
    assert "c0" in result.stdout
    assert "c1" in result.stdout


def test_c0_builds_40_deterministic_views_with_recorded_nonzero_rolls():
    from simdiff_eval.probe_controls import build_c0_specs

    first, first_offsets = build_c0_specs(seed=23)
    second, second_offsets = build_c0_specs(seed=23)
    assert len(first) == 40
    assert [spec.name for spec in first] == [spec.name for spec in second]
    assert first_offsets == second_offsets
    assert len(first_offsets) == 4
    assert len(set(first_offsets)) == 4
    assert all((dx, dy) != (0, 0) for dx, dy in first_offsets)
    assert sum(spec.name == "identity" for spec in first) == 1
    assert {spec.dihedral_g for spec in first} == set(range(8))


def synthetic_c0_predictions() -> pd.DataFrame:
    rows = []
    roll_offsets = [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
    for sim_index in (900, 901):
        theta_true = 0.3 + 0.05 * (sim_index - 900)
        for z_index in range(4):
            base = theta_true + 0.01 * z_index
            for dihedral_g in range(8):
                for roll_state, (dx, dy) in enumerate(roll_offsets):
                    no_roll = roll_state == 0
                    if no_roll and dihedral_g == 0:
                        transform = "identity"
                        family = "identity"
                    elif no_roll:
                        transform = f"dihedral_g{dihedral_g}"
                        family = "dihedral"
                    else:
                        transform = f"dihedral_g{dihedral_g}__roll_dx{dx}_dy{dy}"
                        family = "roll"
                    rows.append(
                        {
                            "transform": transform,
                            "transform_family": family,
                            "k_cut": None,
                            "k_cut_over_knyq": None,
                            "window": None,
                            "dihedral_g": dihedral_g,
                            "roll_dx": dx,
                            "roll_dy": dy,
                            "sim_index": sim_index,
                            "z_index": z_index,
                            "parameter": "Omega_m",
                            "theta_true": theta_true,
                            "theta_pred": base + 0.001 * dihedral_g + 0.01 * roll_state,
                            "out_of_range_fraction": 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_c0_reports_separate_families_and_within_simulation_baseline():
    from simdiff_eval.probe_controls import c0_symmetry_summary

    report = c0_symmetry_summary(synthetic_c0_predictions(), n_boot=50, seed=31)
    assert {row["family"] for row in report["per_slice"]} == {"dihedral", "roll"}
    assert all(np.isfinite(row["std_over_within_sim_std"]) for row in report["per_slice"])
    assert (
        report["family_summary"]["roll"]["median_std_ratio"]
        > report["family_summary"]["dihedral"]["median_std_ratio"]
    )
    assert (
        report["baseline"]["definition"]
        == "identity Omega_m spread across z-slices within each simulation"
    )


def test_combined_c0_c1_predictions_produce_uncontaminated_c0_summary():
    from evaluate_probe_transform_controls import build_requested_specs
    from simdiff_eval.probe_controls import c0_symmetry_summary, evaluate_transform_specs

    images, theta_raw, sim_index, z_index = synthetic_probe_inputs()
    specs, _ = build_requested_specs(
        ["c0", "c1"],
        roll_seed=23,
        k_cuts=[2.0, 3.0],
    )
    predictions = evaluate_transform_specs(
        images,
        theta_raw,
        sim_index,
        z_index,
        FakeEncoder(),
        specs,
        batch_size=4,
    )
    report = c0_symmetry_summary(predictions, n_boot=10, seed=31)
    assert {row["family"] for row in report["per_slice"]} == {
        "dihedral",
        "roll",
    }
    assert len(report["per_slice"]) == len(images) * 2


def test_c4_split_is_deterministic_disjoint_and_complete():
    from simdiff_eval.probe_controls import deterministic_cosmology_split

    heldout = np.arange(900, 932)
    derive_a, evaluate_a = deterministic_cosmology_split(heldout, seed=41)
    derive_b, evaluate_b = deterministic_cosmology_split(heldout, seed=41)
    np.testing.assert_array_equal(derive_a, derive_b)
    np.testing.assert_array_equal(evaluate_a, evaluate_b)
    assert len(derive_a) == len(evaluate_a) == 16
    assert set(derive_a).isdisjoint(evaluate_a)
    assert set(derive_a) | set(evaluate_a) == set(heldout)


def test_gaussian_fit_recovers_known_smoothing_scale():
    from simdiff_eval.probe_controls import fit_gaussian_smoothing

    k = np.linspace(1.0, 20.0, 25)
    expected_sigma = 0.08
    ratio = np.exp(-(expected_sigma * k) ** 2)
    fitted = fit_gaussian_smoothing(k, ratio)
    assert fitted == pytest.approx(expected_sigma, rel=1e-3)


def test_measured_transfer_builds_finite_real_degraded_maps():
    from simdiff_eval.probe_controls import power_ratio_transfer
    from simdiff_eval.probe_transforms import transfer_transform

    rng = np.random.default_rng(51)
    real = rng.normal(size=(6, 1, 16, 16)).astype(np.float32)
    generated = (0.7 * real + 0.1 * rng.normal(size=real.shape)).astype(np.float32)
    k, real_mean, generated_mean, ratio, transfer = power_ratio_transfer(
        real,
        generated,
        nbins=8,
    )
    degraded, diagnostics = transfer_transform(k, transfer)(real)
    assert degraded.shape == real.shape
    assert degraded.dtype == np.float32
    assert not np.iscomplexobj(degraded)
    assert np.isfinite(degraded).all()
    assert np.isfinite(real_mean).all()
    assert np.isfinite(generated_mean).all()
    assert np.isfinite(ratio).all()
    assert diagnostics["out_of_range_fraction"] >= 0.0


def test_c4_same_size_runs_have_distinct_metric_groups():
    from evaluate_probe_degradation_control import c4_transform_names
    from simdiff_eval.probe_controls import (
        TransformSpec,
        aggregate_prediction_table,
        evaluate_transform_specs,
    )
    from simdiff_eval.probe_transforms import get_transform

    images, theta_raw, sim_index, z_index = synthetic_probe_inputs()
    base = evaluate_transform_specs(
        images,
        theta_raw,
        sim_index,
        z_index,
        FakeEncoder(),
        [TransformSpec("identity", "identity", get_transform("identity"))],
        batch_size=4,
    )
    tables = []
    for run_name in ("run_a", "run_b"):
        table = base.copy()
        table["transform"] = c4_transform_names(run_name, 64)["measured"]
        table["transform_family"] = "transfer"
        tables.append(table)
    report = aggregate_prediction_table(
        pd.concat(tables, ignore_index=True),
        n_boot=10,
        seed=53,
    )
    omega_slice = [
        row
        for row in report["metrics"]
        if row["parameter"] == "Omega_m" and row["grain"] == "per_slice"
    ]
    assert len(omega_slice) == 2
    assert {row["n"] for row in omega_slice} == {len(images)}
    assert len({row["transform"] for row in omega_slice}) == 2


def test_generated_cosmology_subset_preserves_sample_target_pairing():
    from simdiff_eval.probe_controls import subset_generated_cosmologies

    heldout = np.array([900, 901, 902])
    samples = np.arange(6 * 4, dtype=np.float32).reshape(6, 1, 2, 2)
    theta = np.arange(3 * 6, dtype=np.float32).reshape(3, 6)
    selected, repeated_theta, sim_index, sample_index = subset_generated_cosmologies(
        samples,
        theta,
        heldout,
        samples_per_cosmology=2,
        selected_simulations=np.array([900, 902]),
    )
    np.testing.assert_array_equal(selected, samples[[0, 1, 4, 5]])
    np.testing.assert_array_equal(repeated_theta, theta[[0, 0, 2, 2]])
    np.testing.assert_array_equal(sim_index, [900, 900, 902, 902])
    np.testing.assert_array_equal(sample_index, [0, 1, 0, 1])


def test_c4_limitation_prevents_overreading_two_point_negative_result():
    from simdiff_eval.probe_controls import C4_LIMITATION

    assert "two-point" in C4_LIMITATION
    assert "one-point PDF" in C4_LIMITATION
    assert "higher-order" in C4_LIMITATION
    assert "only" in C4_LIMITATION


def test_degradation_control_cli_help_is_import_safe():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_probe_degradation_control.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "degraded real" in result.stdout
    assert "--split-seed" in result.stdout
    assert "--manifest" in result.stdout


def test_c1_suite_has_all_required_arms_windows_and_cutoffs():
    from simdiff_eval.probe_controls import DEFAULT_K_CUTS, build_c1_specs

    assert DEFAULT_K_CUTS == (
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        24.0,
        32.0,
        40.0,
        52.0,
        64.0,
    )
    specs = build_c1_specs(DEFAULT_K_CUTS)
    names = [spec.name for spec in specs]
    assert names.count("identity") == 1
    assert names.count("fft_roundtrip_null") == 1
    for k_cut in DEFAULT_K_CUTS:
        label = f"{k_cut:g}"
        for arm in ("lowpass", "highpass"):
            for window in ("sharp", "hann"):
                assert f"{arm}_kcut{label}_{window}" in names
    assert len(specs) == 42


def test_c1_summary_reports_three_curves_at_two_grains():
    from simdiff_eval.probe_controls import (
        TransformSpec,
        c1_scale_cut_summary,
        evaluate_transform_specs,
    )
    from simdiff_eval.probe_transforms import get_transform

    images, theta_raw, sim_index, z_index = synthetic_probe_inputs()
    specs = [
        TransformSpec("identity", "identity", get_transform("identity")),
        TransformSpec(
            "lowpass_kcut2_sharp",
            "lowpass",
            get_transform("lowpass_kcut2_sharp"),
            k_cut=2.0,
            window="sharp",
        ),
        TransformSpec(
            "highpass_kcut2_sharp",
            "highpass",
            get_transform("highpass_kcut2_sharp"),
            k_cut=2.0,
            window="sharp",
        ),
    ]
    table = evaluate_transform_specs(
        images,
        theta_raw,
        sim_index,
        z_index,
        FakeEncoder(),
        specs,
        batch_size=4,
    )
    report = c1_scale_cut_summary(table, n_boot=50, seed=47)
    required = {
        "rmse",
        "bias",
        "slope",
        "rmse_ci_low",
        "rmse_ci_high",
        "bias_ci_low",
        "bias_ci_high",
        "slope_ci_low",
        "slope_ci_high",
        "grain",
        "transform_family",
        "k_cut",
        "k_cut_over_knyq",
        "window",
        "out_of_range_fraction",
    }
    assert all(required.issubset(row) for row in report["curves"])
    assert {row["grain"] for row in report["curves"]} == {
        "per_slice",
        "per_cosmology",
    }
    assert {row["transform_family"] for row in report["curves"]} == {
        "identity",
        "lowpass",
        "highpass",
    }


def test_combined_c0_c1_suite_deduplicates_identity():
    from scripts.evaluate_probe_transform_controls import build_requested_specs

    specs, roll_offsets = build_requested_specs(
        ["c0", "c1"],
        roll_seed=59,
        k_cuts=[4.0, 64.0],
    )
    names = [spec.name for spec in specs]
    assert names.count("identity") == 1
    assert len(roll_offsets) == 4
    assert "fft_roundtrip_null" in names
    assert "lowpass_kcut4_hann" in names
    assert any(name.startswith("dihedral_g7__roll") for name in names)


def test_unrequested_control_summaries_are_removed(tmp_path):
    from evaluate_probe_transform_controls import remove_unrequested_summaries

    c0 = tmp_path / "c0_symmetry_summary.json"
    c1 = tmp_path / "c1_scale_cut_summary.json"
    c0.write_text("stale c0")
    c1.write_text("current c1")
    remove_unrequested_summaries(tmp_path, {"c1"})
    assert not c0.exists()
    assert c1.exists()
