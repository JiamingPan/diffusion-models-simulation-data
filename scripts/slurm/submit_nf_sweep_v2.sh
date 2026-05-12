#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}

cd "${PROJECT_DIR}"
bash scripts/setup_cosmodiff_main.sh
python scripts/prepare_nf_sweep_v2_configs.py --project-dir "${PROJECT_DIR}"

train_job=$(sbatch --parsable scripts/slurm/train_nf_sweep_v2_array.sbatch)
echo "Submitted v2 training array: ${train_job}"

sample_job=$(sbatch --parsable --dependency=afterok:${train_job} scripts/slurm/sample_nf_sweep_v2_array.sbatch)
echo "Submitted v2 sampling array: ${sample_job}"
