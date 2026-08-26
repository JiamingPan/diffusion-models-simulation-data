from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _metrics_module():
    return importlib.import_module("simdiff_eval.paper_nearest_training")


def _plotting_module():
    return importlib.import_module("plot_nf_conditional_bias_paper_figures")


def _write_reference_config(tmp_path: Path) -> Path:
    raw = np.linspace(1.0, 4.0, 2 * 4 * 8 * 8, dtype=np.float32).reshape(2, 4, 8, 8)
    raw_path = tmp_path / "raw.npy"
    np.save(raw_path, raw)
    config = {
        "data": {
            "img_path": [str(raw_path)],
            "img_read_fn": "npy_read_fn",
            "reshape": "2d",
            "zthin": 1,
            "n_samples": [1],
            "seed": None,
            "normalization": "tanh",
            "norm_kwargs": {
                "center": None,
                "xmax": None,
                "alpha": 0.8,
                "beta": 10.0,
                "delta": 1.0,
                "gamma": 1.0,
                "sigma": 1.5,
            },
            "transform": ["log"],
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_exact_reference_summary_checks_manifest_count_and_nyquist(tmp_path):
    metrics = _metrics_module()
    config_path = _write_reference_config(tmp_path)
    generated = np.linspace(0.1, 0.9, 6 * 8 * 8, dtype=np.float32).reshape(6, 1, 8, 8)

    summary = metrics.summarize_exact_training_reference(
        config_path,
        expected_slices=4,
        generated=generated,
        nbins=12,
        k_max=4.0,
    )

    assert summary["n_training_slices"] == 4
    assert summary["n_generated"] == 6
    assert summary["nearest_training_index"] in range(4)
    assert np.max(summary["k_bins"]) <= 4.0
    assert len(summary["pk_ratio"]) == len(summary["k_bins"])
    expected_mae = np.mean(np.abs(np.log10(np.clip(summary["pk_ratio"], 1e-30, None))))
    assert summary["pk_log10_mae"] == pytest.approx(expected_mae)

    with pytest.raises(RuntimeError, match="manifest/config slice-count mismatch"):
        metrics.summarize_exact_training_reference(
            config_path,
            expected_slices=5,
            generated=generated,
            nbins=12,
            k_max=4.0,
        )


def test_normalized_sscd_frechet_uses_equal_sample_counts():
    metrics = _metrics_module()
    rng = np.random.default_rng(7)
    heldout = rng.normal(size=(24, 8))
    generated = rng.normal(loc=0.2, size=(10, 8))

    result = metrics.normalized_sscd_frechet(
        heldout,
        generated,
        seed=123,
        max_components=6,
    )

    assert result["n_generated"] == 10
    assert result["n_real_split"] == 10
    assert result["pca_rank"] == 6
    assert np.isfinite(result["sscd_frechet_normalized"])
    assert result["sscd_frechet_normalized"] >= 0.0

    reduced = metrics.normalized_sscd_frechet(heldout[:19], generated, seed=123)
    assert reduced["n_generated"] == reduced["n_real_split"] == 9

    with pytest.raises(ValueError, match="two equal real splits"):
        metrics.normalized_sscd_frechet(heldout[:3], generated, seed=123)


def test_missing_sscd_cache_fails_closed_with_pattern(tmp_path):
    metrics = _metrics_module()
    with pytest.raises(FileNotFoundError) as excinfo:
        metrics.resolve_sscd_embedding_cache(
            tmp_path,
            run_name="nf_fig2_u128_d2p06_noaug_200k",
            kind="generated",
            sample_label="dpm50",
            seed=123,
        )
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "nf_fig2_u128_d2p06_noaug_200k_generated_dpm50_seed123_*.pt" in message


def _figure_panels() -> list[dict[str, object]]:
    panels: list[dict[str, object]] = []
    x = np.linspace(-1.0, 1.0, 128)
    base = np.outer(np.sin(np.pi * x), np.cos(np.pi * x)).astype(np.float32)
    for power in range(6, 12):
        panels.append(
            {
                "dataset_tag": f"d2p{power:02d}",
                "dataset_size": 2**power,
                "generated_image": base + 0.01 * power,
                "nearest_training_image": base,
                "cos_max": 0.97 - 0.02 * (power - 6),
                "k_bins": np.linspace(2.0, 64.0, 20),
                "pk_ratio": np.linspace(0.82, 1.15 + 0.03 * (power - 6), 20),
                "pk_log10_mae": 0.05 + 0.01 * (power - 6),
                "sscd_frechet_normalized": 0.9 + 0.1 * (power - 6),
                "n_generated": 512,
                "n_real_split": 512,
                "config_path": f"local/nf_generalize_fig2/configs/u128_d2p{power:02d}.yaml",
            }
        )
    return panels


def test_three_row_figure_and_csv_contract(tmp_path):
    plotting = _plotting_module()
    panels = _figure_panels()
    figure = plotting.build_nearest_training_figure(panels)
    try:
        assert figure.get_size_inches()[0] == pytest.approx(6.75)
        assert len(figure.axes) == 18
        image_axes = np.asarray(figure.axes[:12], dtype=object).reshape(2, 6)
        spectrum_axes = figure.axes[12:]
        assert [axis.get_title() for axis in image_axes[0]] == [
            rf"$2^{{{power}}}$" for power in range(6, 12)
        ]
        assert "Generated" in image_axes[0, 0].get_ylabel()
        assert "Closest training" in image_axes[1, 0].get_ylabel()
        assert spectrum_axes[0].get_ylabel() == r"$R(k)=P_{\rm gen}/P_{\rm real}$"
        assert all(axis.get_ylim() == pytest.approx(spectrum_axes[0].get_ylim()) for axis in spectrum_axes)
        assert all(max(line.get_xdata()) <= 64.0 for axis in spectrum_axes for line in axis.lines)
        assert "F=0.90" in "\n".join(text.get_text() for text in image_axes[0, 0].texts)
        assert "cos=0.97" in "\n".join(text.get_text() for text in image_axes[1, 0].texts)

        pdf_path = tmp_path / "nearest_training_u128.pdf"
        csv_path = tmp_path / "nearest_training_u128.csv"
        dimensions, table = plotting.export_nearest_training_outputs(
            panels,
            pdf_path,
            csv_path,
        )
    finally:
        plt.close(figure)

    assert dimensions[0] == pytest.approx(6.75)
    assert pdf_path.read_bytes().startswith(b"%PDF")
    saved = pd.read_csv(csv_path)
    expected_columns = [
        "dataset_tag",
        "dataset_size",
        "cos_max",
        "pk_log10_mae",
        "sscd_frechet_normalized",
        "n_generated",
        "n_real_split",
        "config_path",
    ]
    assert list(saved.columns) == expected_columns
    assert len(saved) == 6
    pd.testing.assert_frame_equal(saved, table, check_dtype=False)


def test_authoritative_slope_annotation_keeps_interval():
    plotting = _plotting_module()
    text = plotting.format_slope_with_interval(0.286266, 0.229859, 0.337462)
    assert text == r"$0.286^{+0.051}_{-0.056}$"
