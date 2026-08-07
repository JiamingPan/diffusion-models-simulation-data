#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
LOG_DIR=${PROJECT_DIR}/logs/nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

job_id=$(sbatch -A "${ACCOUNT}" --parsable \
  scripts/slurm/sample_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit_array.sbatch)

echo "controlled sampler audit job: ${job_id}"
echo "DPM50 baseline is reused; this job creates DPM100, DPM200, and DDPM500 archives."
echo "All methods use the same fresh 300k checkpoints, seed 123, and 512 samples."
echo "The array runs at most two one-GPU tasks concurrently; each task has 24 hours."

