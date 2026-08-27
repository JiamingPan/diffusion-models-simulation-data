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
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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
    assert len(summary["generated_pk_mean"]) == len(summary["k_bins"])
    assert len(summary["real_pk_median"]) == len(summary["k_bins"])
    assert summary["real_pk_percentiles"].shape == (4, len(summary["k_bins"]))
    q025, q16, q84, q975 = summary["real_pk_percentiles"]
    assert np.all(q025 <= q16)
    assert np.all(q16 <= summary["real_pk_median"])
    assert np.all(summary["real_pk_median"] <= q84)
    assert np.all(q84 <= q975)
    np.testing.assert_allclose(
        summary["pk_ratio"],
        summary["generated_pk_mean"] / summary["real_pk_mean"],
    )
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
    assert result["n_heldout_real_available"] == 24
    assert result["reference_kind"] == "heldout_real"
    assert result["pca_rank"] == 6
    assert np.isfinite(result["sscd_frechet_normalized"])
    assert result["sscd_frechet_normalized"] >= 0.0
    assert result["sscd_frechet_normalized"] == pytest.approx(
        result["generated_to_real_frechet"] / result["real_split_frechet"]
    )

    with pytest.raises(ValueError, match="matched-n heldout-real baseline"):
        metrics.normalized_sscd_frechet(heldout[:19], generated, seed=123)


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
    for power in (6, 8, 10, 12, 15):
        panels.append(
            {
                "dataset_tag": f"d2p{power:02d}",
                "dataset_size": 2**power,
                "generated_image": base + 0.01 * power,
                "nearest_training_image": base,
                "cos_max": 0.97 - 0.02 * (power - 6),
                "k_bins": np.linspace(2.0, 64.0, 20),
                "pk_ratio": np.linspace(0.82, 1.15 + 0.03 * (power - 6), 20),
                "generated_pk_mean": np.geomspace(4.0, 0.05, 20)
                * (1.0 + 0.015 * (power - 6)),
                "real_pk_mean": np.geomspace(4.1, 0.052, 20),
                "real_pk_median": np.geomspace(4.0, 0.05, 20),
                "real_pk_percentiles": np.stack(
                    (
                        np.geomspace(2.0, 0.025, 20),
                        np.geomspace(3.0, 0.0375, 20),
                        np.geomspace(5.0, 0.0625, 20),
                        np.geomspace(6.5, 0.08125, 20),
                    )
                ),
                "pk_log10_mae": 0.05 + 0.01 * (power - 6),
                "sscd_frechet_normalized": 0.9 + 0.1 * (power - 6),
                "generated_to_heldout_real_frechet": 1.8 + 0.2 * (power - 6),
                "heldout_real_split_frechet": 2.0,
                "n_generated": 512,
                "n_real_split": 512,
                "n_heldout_real_available": 1024,
                "sscd_reference_kind": "heldout_real",
                "generated_cache_path": f"cache/u128_d2p{power:02d}_generated.pt",
                "heldout_real_cache_path": f"cache/u128_d2p{power:02d}_heldout.pt",
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
        assert len(figure.axes) == 15
        image_axes = np.asarray(figure.axes[:10], dtype=object).reshape(2, 5)
        spectrum_axes = figure.axes[10:]
        assert [axis.get_title() for axis in image_axes[0]] == [
            rf"$2^{{{power}}}$" for power in (6, 8, 10, 12, 15)
        ]
        assert "Generated" in image_axes[0, 0].get_ylabel()
        assert "Closest training" in image_axes[1, 0].get_ylabel()
        assert spectrum_axes[0].get_ylabel() == r"$P(k)$"
        assert all(axis.get_ylim() == pytest.approx(spectrum_axes[0].get_ylim()) for axis in spectrum_axes)
        assert all(axis.get_yscale() == "log" for axis in spectrum_axes)
        expected_k_max = 2.0 * np.pi * 64.0 / 25.0
        assert all(max(line.get_xdata()) <= expected_k_max + 1.0e-9 for axis in spectrum_axes for line in axis.lines)
        assert all(
            axis.get_xlabel() == r"$k\,[h\,\mathrm{Mpc}^{-1}]$"
            for axis in spectrum_axes
        )
        assert all(
            [tick.get_text() for tick in axis.get_xticklabels()] == ["0", "8", "16"]
            for axis in spectrum_axes
        )
        assert all(len(axis.collections) == 2 for axis in spectrum_axes)
        assert all(len(axis.lines) == 2 for axis in spectrum_axes)
        assert all(
            {line.get_label() for line in axis.lines}
            == {"Generated mean", "Real median"}
            for axis in spectrum_axes
        )
        assert spectrum_axes[0].get_legend_handles_labels()[1] == [
            "Real median",
            "Generated mean",
        ]
        assert all(
            not any(np.allclose(line.get_ydata(), 1.0) for line in axis.lines)
            for axis in spectrum_axes
        )
        generated_annotation = "\n".join(
            text.get_text() for text in image_axes[0, 0].texts
        )
        assert "FD_SSCD=0.90" in generated_annotation
        assert "F=0.90" not in generated_annotation
        assert "cos=0.97" in "\n".join(text.get_text() for text in image_axes[1, 0].texts)
        assert all(
            image.get_interpolation() == "none"
            for axis in image_axes.flat
            for image in axis.images
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        assert all(
            axis.title.get_window_extent(renderer).y1 <= figure.bbox.y1
            for axis in image_axes[0]
        )

        pdf_path = tmp_path / "nearest_training_u128.pdf"
        preview_path = tmp_path / "nearest_training_u128_preview.png"
        csv_path = tmp_path / "nearest_training_u128.csv"
        caption_path = tmp_path / "nearest_training_u128_caption.tex"
        dimensions, table = plotting.export_nearest_training_outputs(
            panels,
            pdf_path,
            csv_path,
            preview_path=preview_path,
            caption_path=caption_path,
        )
    finally:
        plt.close(figure)

    assert dimensions[0] == pytest.approx(6.75)
    assert pdf_path.read_bytes().startswith(b"%PDF")
    preview = plt.imread(preview_path)
    assert preview.shape[1] >= 2000
    assert preview.shape[0] >= 900
    saved = pd.read_csv(csv_path)
    expected_columns = [
        "dataset_size",
        "cos_max",
        "fd_sscd",
        "fd_sscd_real_real_baseline",
        "n_generated",
        "n_reference",
        "config_path",
    ]
    assert list(saved.columns) == expected_columns
    assert len(saved) == 5
    pd.testing.assert_frame_equal(saved, table, check_dtype=False)
    caption = caption_path.read_text()
    assert r"$\mathrm{FD}_{\mathrm{SSCD}}$" in caption
    assert "lower is better" in caption
    assert "2.000" in caption
    assert "two disjoint halves of the held-out real set" in caption
    assert "512 fields per half" in caption
    assert "exact training subset" in caption
    assert "16th--84th" in caption
    assert "2.5th--97.5th" in caption
    assert "mean generated spectrum" in caption
    assert "median and percentile bands" in caption
    assert "exact training subset" in caption
    assert r"$R(k)\simeq1$" not in caption
    assert "held-out real fields" in caption


def test_caption_reports_column_specific_real_real_baselines():
    plotting = _plotting_module()
    table = plotting.nearest_training_table(_figure_panels())
    table.loc[table.index[-1], "fd_sscd_real_real_baseline"] = 2.2
    caption = plotting.nearest_training_caption(table)

    assert "column-specific" in caption
    assert r"$N_{2D}=2^{6}$: 2.000" in caption
    assert r"$N_{2D}=2^{15}$: 2.200" in caption
    assert "corresponding run's training normalization" in caption


def test_requested_unet128_columns_fail_closed_when_manifest_run_is_missing(tmp_path):
    plotting = _plotting_module()
    rows = [
        {
            "arch": "u128",
            "dataset_tag": f"d2p{power:02d}",
            "dataset_size": 2**power,
            "run_name": f"u128_d2p{power:02d}",
            "config": f"configs/u128_d2p{power:02d}.yaml",
            "sample_path": f"samples/u128_d2p{power:02d}_{{sample_label}}_seed{{seed}}.npz",
        }
        for power in (6, 8, 10, 12)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(__import__("json").dumps({"rows": rows}))

    with pytest.raises(ValueError, match=r"missing U-Net-128 runs.*32768"):
        plotting.build_nearest_training_panels(
            tmp_path,
            manifest,
            tmp_path / "cache",
        )


def test_authoritative_slope_annotation_keeps_interval():
    plotting = _plotting_module()
    text = plotting.format_slope_with_interval(0.286266, 0.229859, 0.337462)
    assert text == r"$0.286^{+0.051}_{-0.056}$"
