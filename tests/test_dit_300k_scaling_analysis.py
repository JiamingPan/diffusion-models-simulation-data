from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dit_300k_scaling_analysis import (
    build_historical_unet_metric_table,
    build_mixed_dit_metric_table,
    expected_dataset_tags,
    interpolate_n50,
    normalize_generalization_table,
    require_exact_dataset_sweep,
    summarize_n50,
    validate_sample_archive_metadata,
)


def sweep_table(*, arch: str = "dit_l16") -> pd.DataFrame:
    powers = list(range(6, 16))
    return pd.DataFrame(
        {
            "arch": arch,
            "dataset_tag": [f"d2p{power:02d}" for power in powers],
            "dataset_size": [2**power for power in powers],
            "gen_gl_q95": np.linspace(0.0, 1.0, len(powers)),
        }
    )


def test_expected_dataset_tags_cover_full_sweep():
    assert expected_dataset_tags() == tuple(f"d2p{power:02d}" for power in range(6, 16))


def test_require_exact_dataset_sweep_returns_sorted_architecture_rows():
    table = pd.concat([sweep_table(arch="dit_l16"), sweep_table(arch="dit_l8")])
    table = table.sample(frac=1.0, random_state=4)

    result = require_exact_dataset_sweep(
        table,
        arch="dit_l16",
        value_columns=("gen_gl_q95",),
        context="fresh L16",
    )

    assert result["dataset_tag"].tolist() == list(expected_dataset_tags())
    assert result["dataset_size"].tolist() == [2**power for power in range(6, 16)]
    assert set(result["arch"]) == {"dit_l16"}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "nonfinite"])
def test_require_exact_dataset_sweep_rejects_invalid_contract(mutation: str):
    table = sweep_table()
    if mutation == "missing":
        table = table.iloc[:-1].copy()
    elif mutation == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    elif mutation == "extra":
        extra = table.iloc[[0]].copy()
        extra["dataset_tag"] = "d2p16"
        extra["dataset_size"] = 2**16
        table = pd.concat([table, extra], ignore_index=True)
    elif mutation == "nonfinite":
        table.loc[3, "gen_gl_q95"] = np.nan

    with pytest.raises(ValueError, match="fresh L16"):
        require_exact_dataset_sweep(
            table,
            arch="dit_l16",
            value_columns=("gen_gl_q95",),
            context="fresh L16",
        )


def test_interpolate_n50_finds_clean_log2_crossing():
    result = interpolate_n50(
        dataset_sizes=np.array([64, 128, 256, 512]),
        scores=np.array([0.1, 0.3, 0.7, 0.9]),
    )

    assert result.status == "crossing"
    assert result.interval == (128.0, 256.0)
    assert result.n50 == pytest.approx(2 ** 7.5)


def test_interpolate_n50_reports_left_and_right_censoring():
    left = interpolate_n50([64, 128, 256], [0.6, 0.8, 0.9])
    right = interpolate_n50([64, 128, 256], [0.1, 0.2, 0.4])

    assert left.status == "left_censored"
    assert left.n50 == 64.0
    assert right.status == "right_censored"
    assert right.n50 == 256.0


def test_interpolate_n50_rejects_ambiguous_nonmonotonic_crossings():
    result = interpolate_n50([64, 128, 256, 512], [0.2, 0.7, 0.3, 0.8])

    assert result.status == "ambiguous"
    assert np.isnan(result.n50)
    assert result.crossing_count == 3


def test_normalize_generalization_table_derives_score_tag_and_size():
    table = pd.DataFrame(
        {
            "arch": ["dit_l8"],
            "run_name": ["nf_fig2_dit_l8_d2p06_noaug_200k"],
            "gen_copy_fraction_q95": [0.75],
        }
    )

    result = normalize_generalization_table(table, context="historical")

    assert result.loc[0, "dataset_tag"] == "d2p06"
    assert result.loc[0, "dataset_size"] == 64
    assert result.loc[0, "gen_gl_q95"] == pytest.approx(0.25)


def test_build_mixed_dit_metric_table_uses_only_fresh_l16_rows():
    historical = pd.concat(
        [
            sweep_table(arch="dit_l8"),
            sweep_table(arch="dit_base"),
            sweep_table(arch="dit_l16").assign(gen_gl_q95=0.99),
        ],
        ignore_index=True,
    )
    fresh = sweep_table(arch="dit_l16").assign(gen_gl_q95=0.42)

    result = build_mixed_dit_metric_table(
        historical,
        fresh,
        feature="PCA",
    )

    assert len(result) == 30
    assert set(result["arch"]) == {"dit_l8", "dit_base", "dit_l16"}
    assert set(result.loc[result["arch"] == "dit_l16", "gen_gl_q95"]) == {0.42}
    assert set(result.loc[result["arch"] == "dit_l16", "updates_k"]) == {300}
    assert set(result.loc[result["arch"] != "dit_l16", "updates_k"]) == {200}
    assert set(result.loc[result["arch"] == "dit_l16", "source"]) == {
        "fresh independent 300k v2"
    }


def test_build_mixed_dit_metric_table_rejects_incomplete_fresh_l16():
    historical = pd.concat(
        [sweep_table(arch="dit_l8"), sweep_table(arch="dit_base")],
        ignore_index=True,
    )
    fresh = sweep_table(arch="dit_l16").iloc[:-1]

    with pytest.raises(ValueError, match="fresh independent L16 300k"):
        build_mixed_dit_metric_table(historical, fresh, feature="SSCD")


def test_build_historical_unet_metric_table_requires_all_models_and_sizes():
    table = pd.concat(
        [
            sweep_table(arch="u64"),
            sweep_table(arch="u128"),
            sweep_table(arch="u256"),
        ],
        ignore_index=True,
    )

    result = build_historical_unet_metric_table(table, feature="PCA")

    assert len(result) == 30
    assert set(result["arch"]) == {"u64", "u128", "u256"}
    assert set(result["updates_k"]) == {200}
    assert set(result["source"]) == {"historical fixed 200k"}


def test_build_historical_unet_metric_table_rejects_missing_architecture():
    table = pd.concat(
        [sweep_table(arch="u64"), sweep_table(arch="u128")], ignore_index=True
    )

    with pytest.raises(ValueError, match="historical UNet-256"):
        build_historical_unet_metric_table(table, feature="SSCD")


def test_summarize_n50_preserves_censoring_and_mixed_budget_labels():
    table = pd.concat(
        [
            sweep_table(arch="dit_l8").assign(
                feature="PCA", model_label="DiT-L8 200k", updates_k=200
            ),
            sweep_table(arch="dit_l16").assign(
                feature="PCA", model_label="DiT-L16 fresh 300k", updates_k=300
            ),
        ],
        ignore_index=True,
    )
    table.loc[table["arch"] == "dit_l8", "gen_gl_q95"] = 0.9
    table.loc[table["arch"] == "dit_l16", "gen_gl_q95"] = 0.1

    result = summarize_n50(table)

    statuses = result.set_index("arch")["status"].to_dict()
    assert statuses == {"dit_l8": "left_censored", "dit_l16": "right_censored"}
    assert set(result["updates_k"]) == {200, 300}


def valid_sample_metadata(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "checkpoint-epoch-1234"
    config = tmp_path / "config.yaml"
    return {
        "requested_checkpoint": np.asarray(str(checkpoint)),
        "resolved_checkpoint": np.asarray(str(checkpoint)),
        "config_path": np.asarray(str(config)),
        "scheduler": np.asarray("DPMSolverMultistepScheduler"),
        "num_steps": np.asarray(50),
        "seed": np.asarray(123),
        "samples": np.zeros((512, 1, 8, 8), dtype=np.float32),
    }


def validate_metadata(metadata: dict[str, object], tmp_path: Path) -> dict[str, object]:
    return validate_sample_archive_metadata(
        metadata,
        expected_checkpoint=tmp_path / "checkpoint-epoch-1234",
        expected_config_path=tmp_path / "config.yaml",
        expected_scheduler="DPMSolverMultistepScheduler",
        expected_num_steps=50,
        expected_seed=123,
        expected_samples=512,
    )


def test_validate_sample_archive_metadata_normalizes_valid_archive(tmp_path: Path):
    result = validate_metadata(valid_sample_metadata(tmp_path), tmp_path)

    assert result["num_steps"] == 50
    assert result["seed"] == 123
    assert result["n_generated"] == 512
    assert result["scheduler"] == "DPMSolverMultistepScheduler"


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("requested_checkpoint", "/wrong/requested", "requested checkpoint"),
        ("resolved_checkpoint", "/wrong/resolved", "resolved checkpoint"),
        ("config_path", "/wrong/config.yaml", "config path"),
        ("scheduler", "DDPMScheduler", "scheduler"),
        ("num_steps", 100, "step count"),
        ("seed", 999, "seed"),
    ],
)
def test_validate_sample_archive_metadata_rejects_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
):
    metadata = valid_sample_metadata(tmp_path)
    metadata[field] = np.asarray(replacement)

    with pytest.raises(ValueError, match=message):
        validate_metadata(metadata, tmp_path)


def test_validate_sample_archive_metadata_rejects_wrong_sample_count(tmp_path: Path):
    metadata = valid_sample_metadata(tmp_path)
    metadata["samples"] = np.zeros((511, 1, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="sample count"):
        validate_metadata(metadata, tmp_path)
