#!/bin/bash
#SBATCH --job-name=nf_calfpre
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/precheck_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/precheck_%j.err
#SBATCH -A huterer2

set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
cd "${PROJECT_DIR}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then source "${VENV_PATH}/bin/activate"; fi
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
SAMPLE_SCRIPT="${COSMODIFF_DIR}/scripts/cosmodiff_sample.py"
if [[ ! -f "${SAMPLE_SCRIPT}" ]]; then
  echo "Missing cosmodiff sampler: ${SAMPLE_SCRIPT}" >&2
  exit 1
fi
if ! grep -q -- "--checkpoint_epoch" "${SAMPLE_SCRIPT}"; then
  echo "Sampler does not support the required --checkpoint_epoch pin: ${SAMPLE_SCRIPT}" >&2
  exit 1
fi
"${PYTHON_BIN}" scripts/check_nf_conditional_bias_full_sweep.py --project-dir "${PROJECT_DIR}"
