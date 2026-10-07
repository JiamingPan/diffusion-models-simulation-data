#!/bin/bash
# Submit the fixed-cosmology, varying-multiplicity conditional sweep (4 runs).
# Chain: prepare (CPU) -> train x4 (GPU, 2 concurrent) -> sample x4 -> VGG evaluate -> training-theta x4.
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_fixedC64_multiplicity_200k

prepare=$(sbatch -A "${ACCOUNT}" --parsable scripts/slurm/prepare_nf_conditional_fixedC64_multiplicity.sbatch)
train=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${prepare} --array=0-3%2 --parsable scripts/slurm/train_nf_conditional_fixedC64_multiplicity_array.sbatch)
sample=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${train} --array=0-3%2 --parsable scripts/slurm/sample_nf_conditional_fixedC64_multiplicity_array.sbatch)
evaluate=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${sample} --parsable scripts/slurm/evaluate_nf_conditional_fixedC64_multiplicity.sbatch)
trth=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${train} --array=0-3%2 --parsable scripts/slurm/train_theta_nf_conditional_fixedC64_multiplicity_array.sbatch)

echo "prepare=${prepare}"
echo "train m=1,2,4,8 (N=64,128,256,512)=${train}"
echo "sample held-out=${sample}"
echo "VGG evaluation=${evaluate}"
echo "training-theta sampling+report=${trth}"
echo "At most two one-GPU tasks run concurrently; training tasks have 48 h walltime each."
