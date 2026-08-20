#!/bin/bash
# Submit the full conditional UNet calibration sweep on Great Lakes.

set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_fresh_full_sweep_200k

prepare=$(sbatch -A "${ACCOUNT}" --parsable scripts/slurm/prepare_nf_conditional_bias_full_sweep.sbatch)
precheck=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${prepare} --parsable scripts/slurm/precheck_nf_conditional_bias_full_sweep.sbatch)
train=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${precheck} --array=0-9%2 --parsable scripts/slurm/train_nf_conditional_bias_full_sweep_array.sbatch)
sample=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${train} --array=0-9%2 --parsable scripts/slurm/sample_nf_conditional_bias_full_sweep_array.sbatch)
evaluate=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${sample} --parsable scripts/slurm/evaluate_nf_conditional_bias_full_sweep.sbatch)
audit=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${evaluate} --parsable scripts/slurm/audit_nf_conditional_bias_full_sweep.sbatch)

echo "prepare=${prepare}"
echo "fresh/config precheck=${precheck}"
echo "train all ten fresh sizes=${train}"
echo "sample all ten sizes=${sample}"
echo "VGG evaluation and plots=${evaluate}"
echo "final audit=${audit}"
echo "At most two one-GPU tasks run concurrently. Training tasks have 48 hours each."
