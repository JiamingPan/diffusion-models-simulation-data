from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/slurm/sample_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit_array.sbatch"
SUBMIT = ROOT / "scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit.sh"


def test_sampler_audit_array_is_complete_bounded_and_long_enough():
    source = SBATCH.read_text()
    assert "#SBATCH --array=0-29%2" in source
    assert "#SBATCH --time=24:00:00" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --mem=80gb" in source


def test_sampler_audit_uses_three_new_controlled_specs():
    source = SBATCH.read_text()
    required = (
        "dpm100_fresh300k_v2",
        "dpm200_fresh300k_v2",
        "ddpm500_fresh300k_v2",
        "DPMSolverMultistepScheduler:100",
        "DPMSolverMultistepScheduler:200",
        "DDPMScheduler:500",
    )
    for text in required:
        assert text in source
    assert "dpm50_fresh300k_v2" not in source


def test_sampler_audit_reuses_exact_fresh_checkpoint_and_fixed_sampling_controls():
    source = SBATCH.read_text()
    required = (
        "prepare_nf_generalize_fig2_dit_l16_fresh300k_v2_configs.py",
        "field expected_checkpoint",
        "field config",
        "NUM_SAMPLES=${NUM_SAMPLES:-512}",
        "SEED=${SEED:-123}",
        "CLASS_LABEL=${CLASS_LABEL:-0}",
        "--checkpoint \"${CHECKPOINT_PATH}\"",
        "--config \"${CONFIG_PATH}\"",
        "--scheduler \"${SAMPLER_CLASS}\"",
        "--num-steps \"${SAMPLER_STEPS}\"",
        "--seed \"${SEED}\"",
    )
    for text in required:
        assert text in source


def test_sampler_audit_never_overwrites_existing_archives_by_default():
    source = SBATCH.read_text()
    assert "OVERWRITE=${OVERWRITE:-0}" in source
    assert '[[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" ]]' in source
    assert "Skipping existing controlled sampler archive" in source


def test_sampler_audit_submission_creates_logs_before_sbatch():
    source = SUBMIT.read_text()
    assert "mkdir -p" in source
    assert source.index("mkdir -p") < source.index("sbatch")
    assert "sample_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit_array.sbatch" in source
    assert "DPM50 baseline is reused" in source

