#!/bin/bash
#SBATCH --job-name=nf_bias_sscdeval
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/sscd_eval_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/sscd_eval_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-123}
SAMPLES_PER_COSMOLOGY=${SAMPLES_PER_COSMOLOGY:-64}
BOOTSTRAP=${BOOTSTRAP:-1000}
SSCD_DEVICE=${SSCD_DEVICE:-cuda}
SSCD_BATCH_SIZE=${SSCD_BATCH_SIZE:-16}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"${PYTHON_BIN}" scripts/evaluate_nf_conditional_bias_sscd_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --seed "${SEED}" \
  --samples-per-cosmology "${SAMPLES_PER_COSMOLOGY}" \
  --embedding-batch-size "${SSCD_BATCH_SIZE}" \
  --bootstrap "${BOOTSTRAP}" \
  --device "${SSCD_DEVICE}"
