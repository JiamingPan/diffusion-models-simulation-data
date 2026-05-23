#!/bin/bash
#SBATCH --job-name=nf_gen_cmp
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_nick_data/sampler_compare_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_nick_data/sampler_compare_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-123}
DDPM_LABEL=${DDPM_LABEL:-raw_train_full}
DPM_LABEL=${DPM_LABEL:-dpm50_compare}
MAX_SAMPLES=${MAX_SAMPLES:-96}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_nick_data results/nf_generalize_nick_data/sampler_compare results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"${PYTHON_BIN}" scripts/compare_nf_generalize_nick_samplers.py \
  --project-dir "${PROJECT_DIR}" \
  --seed "${SEED}" \
  --ddpm-label "${DDPM_LABEL}" \
  --dpm-label "${DPM_LABEL}" \
  --max-samples "${MAX_SAMPLES}" \
  --output-dir "${PROJECT_DIR}/results/nf_generalize_nick_data/sampler_compare"
