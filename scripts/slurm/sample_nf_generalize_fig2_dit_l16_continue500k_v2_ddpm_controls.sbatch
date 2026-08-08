#!/bin/bash
#SBATCH --job-name=nf_l16_ddpm
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/ddpm_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/ddpm_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2
SAMPLE_ROOT=${SAMPLE_ROOT:-${PROJECT_DIR}/results/${SWEEP}/samples}
SAMPLER_CLASS=DDPMScheduler
SAMPLER_STEPS=500
NUM_SAMPLES=512
BATCH_SIZE=8
IMAGE_SIZE=128
SEED=123
CLASS_LABEL=0
OVERWRITE=${OVERWRITE:-0}
CONTROLS=(d2p08:0 d2p08:5 d2p11:0 d2p11:5)

cd "${PROJECT_DIR}"
mkdir -p "logs/${SWEEP}" "${SAMPLE_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

IFS=: read -r DATASET_TAG STAGE <<< "${CONTROLS[$SLURM_ARRAY_TASK_ID]}"
field() {
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --dataset-tag "${DATASET_TAG}" \
    --stage "$1" \
    --print-field "$2"
}

RUN_NAME=$(field 1 run_name)
if (( STAGE == 0 )); then
  CONFIG_VALUE=$(field 1 source_config)
  CHECKPOINT_PATH=$(field 1 source_checkpoint)
  SAMPLE_LABEL=ddpm500_source_300k
else
  CONFIG_VALUE=$(field 5 config)
  CHECKPOINT_PATH=$(field 5 expected_checkpoint)
  SAMPLE_LABEL=ddpm500_cont_500k
fi
if [[ "${CONFIG_VALUE}" = /* ]]; then
  CONFIG_PATH=${CONFIG_VALUE}
else
  CONFIG_PATH=${PROJECT_DIR}/${CONFIG_VALUE}
fi
OUTPUT_PATH=${SAMPLE_ROOT}/${RUN_NAME}_seed${SEED}_${SAMPLE_LABEL}.npz

if [[ ! -f "${CONFIG_PATH}" || ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Missing exact DDPM control input: config=${CONFIG_PATH} checkpoint=${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" ]]; then
  echo "Validating existing DDPM control before reuse: ${OUTPUT_PATH}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/validate_nf_generalize_fig2_dit_sample.py" \
    --sample "${OUTPUT_PATH}" --sample-root "${SAMPLE_ROOT}" \
    --sample-label "${SAMPLE_LABEL}" --requested-checkpoint "${CHECKPOINT_PATH}" \
    --scheduler "${SAMPLER_CLASS}" --requested-steps "${SAMPLER_STEPS}"
  exit 0
fi

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for DiT sampling')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
  --checkpoint "${CHECKPOINT_PATH}" --config "${CONFIG_PATH}" \
  --output "${OUTPUT_PATH}" --num-samples 1 --batch-size 1 \
  --image-size "${IMAGE_SIZE}" --scheduler "${SAMPLER_CLASS}" \
  --num-steps "${SAMPLER_STEPS}" --seed "${SEED}" \
  --class-label "${CLASS_LABEL}" --device cpu --preflight-only

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
  --checkpoint "${CHECKPOINT_PATH}" --config "${CONFIG_PATH}" \
  --output "${OUTPUT_PATH}" --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" --image-size "${IMAGE_SIZE}" \
  --scheduler "${SAMPLER_CLASS}" --num-steps "${SAMPLER_STEPS}" \
  --seed "${SEED}" --class-label "${CLASS_LABEL}" --device cuda

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/validate_nf_generalize_fig2_dit_sample.py" \
  --sample "${OUTPUT_PATH}" --sample-root "${SAMPLE_ROOT}" \
  --sample-label "${SAMPLE_LABEL}" --requested-checkpoint "${CHECKPOINT_PATH}" \
  --scheduler "${SAMPLER_CLASS}" --requested-steps "${SAMPLER_STEPS}"
