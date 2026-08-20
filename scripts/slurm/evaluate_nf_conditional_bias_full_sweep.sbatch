#!/bin/bash
#SBATCH --job-name=nf_calfeval
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=08:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/evaluate_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/evaluate_%j.err
#SBATCH -A huterer2

set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
PYTHON_BIN=${PYTHON_BIN:-python}
VGG_ENCODER=${VGG_ENCODER:-results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz}
OUTPUT_DIR=${OUTPUT_DIR:-results/nf_conditional_bias_fresh_full_sweep_200k/calibration_vgg}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_fresh_full_sweep_200k results/cache/matplotlib "${OUTPUT_DIR}"
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON_BIN}" scripts/evaluate_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/local/nf_conditional_bias_fresh_full_sweep_200k/manifest.json" \
  --encoder-type vgg \
  --vgg-encoder "${VGG_ENCODER}" \
  --vgg-device cpu \
  --seed 123 \
  --samples-per-cosmology 64 \
  --bootstrap 2000 \
  --output-dir "${PROJECT_DIR}/${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/plot_nf_conditional_bias_full_sweep.py \
  --project-dir "${PROJECT_DIR}" \
  --calibration-dir "${PROJECT_DIR}/${OUTPUT_DIR}"
