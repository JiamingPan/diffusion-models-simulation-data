#!/bin/bash
#SBATCH --job-name=nf_bias_prepare
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=01:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/prepare_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/prepare_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
BIAS_DATASET_SIZES=${BIAS_DATASET_SIZES:-128,16384}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_bias_probe/prepared_data}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_probe_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --prepared-data-root "${PREPARED_DATA_ROOT}" \
  --dataset-sizes "${BIAS_DATASET_SIZES}" \
  --write-arrays \
  --print-table
