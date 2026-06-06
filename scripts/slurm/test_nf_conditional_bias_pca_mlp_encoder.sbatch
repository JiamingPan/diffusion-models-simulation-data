#!/bin/bash
#SBATCH --job-name=nf_bias_mlp_test
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=02:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/pca_mlp_test_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/pca_mlp_test_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_SLICES=${TRAIN_SLICES:-32768}
VAL_SLICES=${VAL_SLICES:-4096}
TEST_SLICES_PER_SIM=${TEST_SLICES_PER_SIM:-128}
MLP_HIDDEN_LAYERS=${MLP_HIDDEN_LAYERS:-256,128}
MLP_ALPHA=${MLP_ALPHA:-0.0001}
MLP_LR=${MLP_LR:-0.001}
MLP_MAX_ITER=${MLP_MAX_ITER:-700}
OUTPUT_DIR=${OUTPUT_DIR:-results/nf_conditional_bias_probe/encoder}
MODEL_OUT=${MODEL_OUT:-pca_mlp_encoder.pkl}
ENCODER_OUT=${ENCODER_OUT:-pca_mlp_encoder.npz}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON_BIN}" scripts/test_nf_conditional_bias_pca_mlp_encoder.py \
  --project-dir "${PROJECT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-out "${MODEL_OUT}" \
  --encoder-out "${ENCODER_OUT}" \
  --train-slices "${TRAIN_SLICES}" \
  --val-slices "${VAL_SLICES}" \
  --test-slices-per-sim "${TEST_SLICES_PER_SIM}" \
  --hidden-layers "${MLP_HIDDEN_LAYERS}" \
  --alpha "${MLP_ALPHA}" \
  --learning-rate "${MLP_LR}" \
  --max-iter "${MLP_MAX_ITER}"
