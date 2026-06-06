#!/bin/bash
#SBATCH --job-name=nf_bias_vggenc
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/vgg_encoder_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/vgg_encoder_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
VGG_DEVICE=${VGG_DEVICE:-auto}
VGG_WEIGHTS=${VGG_WEIGHTS:-IMAGENET1K_V1}
VGG_IMAGE_SIZE=${VGG_IMAGE_SIZE:-224}
VGG_POOL=${VGG_POOL:-avgmax}
VGG_BATCH_SIZE=${VGG_BATCH_SIZE:-32}
HEAD_TYPE=${HEAD_TYPE:-mlp}
HEAD_TRAIN_SLICES=${HEAD_TRAIN_SLICES:-32768}
HEAD_VAL_SLICES=${HEAD_VAL_SLICES:-4096}
TEST_SLICES_PER_SIM=${TEST_SLICES_PER_SIM:-128}
MLP_HIDDEN_LAYERS=${MLP_HIDDEN_LAYERS:-512,256}
MLP_ALPHA=${MLP_ALPHA:-0.0001}
MLP_LR=${MLP_LR:-0.0003}
MLP_MAX_ITER=${MLP_MAX_ITER:-700}
RIDGE_ALPHA=${RIDGE_ALPHA:-10.0}
ENCODER_OUT=${ENCODER_OUT:-results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz}
MODEL_OUT=${MODEL_OUT:-results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.pkl}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"

"${PYTHON_BIN}" scripts/train_nf_conditional_bias_vgg_encoder.py \
  --project-dir "${PROJECT_DIR}" \
  --device "${VGG_DEVICE}" \
  --weights "${VGG_WEIGHTS}" \
  --image-size "${VGG_IMAGE_SIZE}" \
  --pool "${VGG_POOL}" \
  --embedding-batch-size "${VGG_BATCH_SIZE}" \
  --head-type "${HEAD_TYPE}" \
  --head-train-slices "${HEAD_TRAIN_SLICES}" \
  --head-val-slices "${HEAD_VAL_SLICES}" \
  --test-slices-per-sim "${TEST_SLICES_PER_SIM}" \
  --hidden-layers "${MLP_HIDDEN_LAYERS}" \
  --mlp-alpha "${MLP_ALPHA}" \
  --mlp-lr "${MLP_LR}" \
  --mlp-max-iter "${MLP_MAX_ITER}" \
  --ridge-alpha "${RIDGE_ALPHA}" \
  --encoder-out "${ENCODER_OUT}" \
  --model-out "${MODEL_OUT}"
