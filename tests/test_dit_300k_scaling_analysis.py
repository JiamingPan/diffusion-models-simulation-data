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
    aggregate_physical_batches,
    build_historical_unet_metric_table,
    build_mixed_dit_metric_table,
    checkpoint_metric_candidates,
    evenly_spaced_indices,
    expected_dataset_tags,
    interpolate_n50,
    normalize_generalization_table,
    per_sample_physical_errors,
    prepare_loss_history,
    prepare_stitched_loss_history,
    require_exact_dataset_sweep,
    robust_log_ratio_outliers,
    stage_loss_metrics_from_logs,
    summarize_filtered_power_ratios,
    streaming_nearest_neighbors,
    summarize_n50,
    validate_sampler_endpoint,
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


def test_prepare_loss_history_uses_optimizer_update_axis_and_cycle_average():
    metrics = {
        "epoch_loss": [8.0, 6.0, 4.0, 2.0],
        "optimizer_updates": [2, 4, 6, 8],
    }

    result = prepare_loss_history(
        metrics,
        steps_per_epoch=2,
        target_updates=8,
        restart_updates=4,
    )

    assert result["recorded_updates"] == 8
    assert result["updates"].tolist() == [3.0, 5.0, 7.0]
    assert result["cycle_averaged_loss"].tolist() == [7.0, 5.0, 3.0]
    assert result["tail_median_loss"] == pytest.approx(2.0)


def test_prepare_loss_history_rejects_incomplete_fresh_run():
    metrics = {"epoch_loss": [1.0] * 9}

    with pytest.raises(ValueError, match="recorded 90 optimizer updates"):
        prepare_loss_history(
            metrics,
            steps_per_epoch=10,
            target_updates=100,
            minimum_fraction=0.98,
        )


def test_checkpoint_metric_candidates_selects_only_the_exact_checkpoint(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "checkpoint-epoch-100"
    checkpoint.mkdir(parents=True)
    local_metrics = checkpoint / "metrics.json"
    exact_metrics = checkpoint_root / "metrics_epoch_100.json"
    later_metrics = checkpoint_root / "metrics_epoch_200.json"
    root_metrics = checkpoint_root / "metrics.json"
    for path in (local_metrics, exact_metrics, later_metrics, root_metrics):
        path.write_text("{}")

    assert checkpoint_metric_candidates(checkpoint) == [
        local_metrics,
        exact_metrics,
    ]


def test_stage_loss_metrics_from_logs_reconstructs_exact_epoch_interval(tmp_path: Path):
    first_log = tmp_path / "train_stage100_0.out"
    second_log = tmp_path / "train_stage200_0.out"
    first_log.write_text(
        "Epoch 99 - avg loss: 9.9\n"
        "Epoch 100 - avg loss: 1.0\n"
        "Epoch 101 — avg loss: 0.8\n"
    )
    second_log.write_text(
        "Epoch 102 — avg loss: 6.0e-1\n"
        "Epoch 103 - avg loss: 0.4\n"
        "Epoch 104 - avg loss: 0.2\n"
    )

    metrics, used_paths = stage_loss_metrics_from_logs(
        [second_log, first_log],
        first_epoch=100,
        final_epoch=103,
    )

    assert metrics["epoch_loss"] == pytest.approx([1.0, 0.8, 0.6, 0.4])
    assert used_paths == (first_log, second_log)


def test_stage_loss_metrics_from_logs_rejects_missing_epoch(tmp_path: Path):
    log_path = tmp_path / "train_stage100_0.out"
    log_path.write_text(
        "Epoch 100 - avg loss: 1.0\n"
        "Epoch 102 - avg loss: 0.6\n"
    )

    with pytest.raises(ValueError, match="missing 1 of 3 expected epochs.*101"):
        stage_loss_metrics_from_logs(
            [log_path],
            first_epoch=100,
            final_epoch=102,
        )


def test_prepare_stitched_loss_history_places_stage_local_metrics_on_global_axis():
    result = prepare_stitched_loss_history(
        [
            ({"epoch_loss": [8.0, 6.0]}, 300, 304),
            ({"epoch_loss": [4.0, 2.0]}, 304, 308),
        ],
        steps_per_epoch=2,
        restart_updates=2,
    )

    assert result["updates"].tolist() == [302.0, 304.0, 306.0, 308.0]
    assert result["cycle_averaged_loss"].tolist() == [8.0, 6.0, 4.0, 2.0]
    assert result["start_updates"] == 300
    assert result["recorded_updates"] == 308
    assert result["stage_recorded_updates"].tolist() == [4, 4]


def test_prepare_stitched_loss_history_rejects_incomplete_stage():
    with pytest.raises(ValueError, match="segment 2 recorded 2 optimizer updates"):
        prepare_stitched_loss_history(
            [
                ({"epoch_loss": [8.0, 6.0]}, 300, 304),
                ({"epoch_loss": [4.0]}, 304, 308),
            ],
            steps_per_epoch=2,
            restart_updates=2,
        )


def test_validate_sampler_endpoint_accepts_zero_terminal_sigma():
    result = validate_sampler_endpoint(
        scheduler_class="DPMSolverMultistepScheduler",
        executed_steps=50,
        expected_steps=50,
        final_timestep=20.0,
        terminal_sigma=0.0,
        terminal_sigma_is_zero=True,
        terminal_sigma_verifiable=True,
    )

    assert result == "terminal sigma = 0"


def test_validate_sampler_endpoint_accepts_ddpm_final_timestep_zero_without_sigmas():
    result = validate_sampler_endpoint(
        scheduler_class="DDPMScheduler",
        executed_steps=500,
        expected_steps=500,
        final_timestep=0.0,
        terminal_sigma=np.nan,
        terminal_sigma_is_zero=False,
        terminal_sigma_verifiable=False,
    )

    assert result == "final diffusion timestep = 0"


def test_validate_sampler_endpoint_rejects_truncated_ddpm_schedule():
    with pytest.raises(ValueError, match="did not reach final diffusion timestep 0"):
        validate_sampler_endpoint(
            scheduler_class="DDPMScheduler",
            executed_steps=500,
            expected_steps=500,
            final_timestep=2.0,
            terminal_sigma=np.nan,
            terminal_sigma_is_zero=False,
            terminal_sigma_verifiable=False,
        )


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


def test_evenly_spaced_indices_are_deterministic_and_cover_endpoints():
    result = evenly_spaced_indices(total=512, count=4)

    assert result.tolist() == [0, 170, 341, 511]
    assert len(np.unique(result)) == 4


@pytest.mark.parametrize(
    ("total", "count"),
    [(0, 1), (3, 0), (3, 4)],
)
def test_evenly_spaced_indices_reject_invalid_requests(total: int, count: int):
    with pytest.raises(ValueError):
        evenly_spaced_indices(total=total, count=count)


def test_streaming_nearest_neighbors_searches_every_batch_and_tracks_offsets():
    generated = np.asarray(
        [
            [[[0.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 1.0], [0.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    training_batches = [
        np.asarray(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.8]]],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [[[1.0, 1.0], [0.0, 0.1]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
            ],
            dtype=np.float32,
        ),
    ]

    result = streaming_nearest_neighbors(generated, training_batches)

    assert result["n_training"] == 4
    assert result["nearest_index"].tolist() == [1, 2]
    assert result["mse"].tolist() == pytest.approx([0.01, 0.0025])
    assert result["cosine_similarity"].tolist() == pytest.approx(
        [1.0, 2.0 / np.sqrt(2.01) / np.sqrt(2.0)]
    )
    assert np.array_equal(result["nearest_images"][0], training_batches[0][1])
    assert np.array_equal(result["nearest_images"][1], training_batches[1][0])


def test_streaming_nearest_neighbors_rejects_shape_mismatch_and_empty_reference():
    generated = np.zeros((2, 1, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="no training samples"):
        streaming_nearest_neighbors(generated, [])
    with pytest.raises(ValueError, match="shape"):
        streaming_nearest_neighbors(
            generated,
            [np.zeros((3, 1, 8, 8), dtype=np.float32)],
        )


def test_aggregate_physical_batches_is_invariant_to_batch_partitioning():
    rng = np.random.default_rng(14)
    images = rng.normal(size=(6, 1, 8, 8)).astype(np.float32)
    edges = np.linspace(-4.0, 4.0, 33)

    together = aggregate_physical_batches([images], hist_edges=edges, nbins=5)
    partitioned = aggregate_physical_batches(
        [images[:2], images[2:5], images[5:]],
        hist_edges=edges,
        nbins=5,
    )

    assert together["n_images"] == partitioned["n_images"] == 6
    assert together["n_pixels"] == partitioned["n_pixels"] == 6 * 8 * 8
    assert together["hist"] == pytest.approx(partitioned["hist"])
    assert together["mean_pk"] == pytest.approx(partitioned["mean_pk"])
    assert together["kbins"] == pytest.approx(partitioned["kbins"])


def test_per_sample_physical_errors_are_zero_for_identical_reference_samples():
    rng = np.random.default_rng(5)
    image = rng.uniform(-0.9, 0.9, size=(1, 1, 8, 8)).astype(np.float32)
    edges = np.linspace(-1.0, 1.0, 17)
    reference = aggregate_physical_batches([image], hist_edges=edges, nbins=5)

    result = per_sample_physical_errors(
        image,
        reference_hist=reference["hist"],
        hist_edges=reference["hist_edges"],
        reference_mean_pk=reference["mean_pk"],
        nbins=5,
    )

    assert result["hist_l1"].tolist() == pytest.approx([0.0])
    assert result["pk_log10_mae"].tolist() == pytest.approx([0.0])
    assert result["pk_ratio"][0] == pytest.approx(np.ones(5))


def test_physical_helpers_reject_empty_or_incompatible_inputs():
    edges = np.linspace(-1.0, 1.0, 17)
    with pytest.raises(ValueError, match="no images"):
        aggregate_physical_batches([], hist_edges=edges, nbins=5)
    with pytest.raises(ValueError, match="reference histogram"):
        per_sample_physical_errors(
            np.zeros((2, 1, 8, 8), dtype=np.float32),
            reference_hist=np.zeros(3),
            hist_edges=edges,
            reference_mean_pk=np.ones(5),
            nbins=5,
        )


def test_robust_log_ratio_outliers_flags_only_extreme_sample():
    values = np.asarray([0.91, 0.96, 1.00, 1.04, 1.09, 75.0])

    result = robust_log_ratio_outliers(values, threshold=4.5)

    assert result["outlier_mask"].tolist() == [False, False, False, False, False, True]
    assert result["n_total"] == 6
    assert result["n_flagged"] == 1
    assert result["threshold"] == pytest.approx(4.5)
    assert result["robust_score"][-1] > 4.5


def test_robust_log_ratio_outliers_flags_nothing_when_mad_is_zero():
    result = robust_log_ratio_outliers(np.ones(8), threshold=4.5)

    assert not result["outlier_mask"].any()
    assert result["n_flagged"] == 0
    assert result["scaled_mad"] == pytest.approx(0.0)


def test_robust_log_ratio_outliers_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="strictly positive"):
        robust_log_ratio_outliers([1.0, 0.0, 2.0])


def test_summarize_filtered_power_ratios_reports_original_and_retained_values():
    ratios = np.asarray(
        [
            [0.9, 1.0, 1.1],
            [1.0, 1.1, 1.2],
            [1.1, 1.2, 1.3],
            [9.0, 12.0, 15.0],
        ]
    )

    rows = summarize_filtered_power_ratios(
        ratios,
        outlier_mask=np.asarray([False, False, False, True]),
        bin_indices=(0, 2),
    )

    assert [row["k_bin"] for row in rows] == [0, 2]
    assert {row["n_total"] for row in rows} == {4}
    assert {row["n_kept"] for row in rows} == {3}
    assert rows[0]["original_mean"] == pytest.approx(3.0)
    assert rows[0]["filtered_mean"] == pytest.approx(1.0)
    assert rows[1]["original_median"] == pytest.approx(1.25)
    assert rows[1]["filtered_variance"] == pytest.approx(np.var([1.1, 1.2, 1.3]))
