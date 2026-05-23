#!/bin/bash
#SBATCH --job-name=nf_cls_cmp_an
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_class_conditional_u128/sampler_compare_analyze_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_class_conditional_u128/sampler_compare_analyze_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-123}
NUM_SAMPLES=${NUM_SAMPLES:-96}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_class_conditional_u128 results/nf_class_conditional_u128/sampler_compare results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "Missing class-conditional env: ${VENV_PATH}" >&2
  echo "Run first: bash scripts/setup_nf_class_conditional_env.sh" >&2
  exit 1
fi

set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"${PYTHON_BIN}" scripts/compare_nf_class_conditional_samplers.py \
  --project-dir "${PROJECT_DIR}" \
  --seed "${SEED}" \
  --sample-n "${NUM_SAMPLES}" \
  --output-dir "${PROJECT_DIR}/results/nf_class_conditional_u128/sampler_compare"
