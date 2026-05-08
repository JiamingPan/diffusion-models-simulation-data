#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}

cd "${PROJECT_DIR}"
python scripts/prepare_nf_sweep_configs.py --project-dir "${PROJECT_DIR}"

train_job=$(sbatch --parsable scripts/slurm/train_nf_sweep_array.sbatch)
echo "Submitted training array: ${train_job}"

sample_job=$(sbatch --parsable --dependency=afterok:${train_job} scripts/slurm/sample_nf_sweep_array.sbatch)
echo "Submitted sampling array: ${sample_job}"
