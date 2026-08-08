#!/bin/bash
#SBATCH --job-name=nf_l16_c5s
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --array=0-9%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/sample_stage%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/sample_stage%A_%a.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CONTINUE_STAGE=${CONTINUE_STAGE:?Set CONTINUE_STAGE to 0, 1, 2, 3, 4, or 5}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2
SAMPLE_ROOT=${SAMPLE_ROOT:-${PROJECT_DIR}/results/${SWEEP}/samples}
SAMPLER_CLASS=DPMSolverMultistepScheduler
SAMPLER_STEPS=50
NUM_SAMPLES=512
BATCH_SIZE=8
IMAGE_SIZE=128
SEED=123
CLASS_LABEL=0
OVERWRITE=${OVERWRITE:-0}
DATASET_TAGS=(d2p06 d2p07 d2p08 d2p09 d2p10 d2p11 d2p12 d2p13 d2p14 d2p15)

if (( CONTINUE_STAGE < 0 || CONTINUE_STAGE > 5 )); then
  echo "CONTINUE_STAGE must be between 0 and 5; got ${CONTINUE_STAGE}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "logs/${SWEEP}" "${SAMPLE_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

DATASET_TAG=${DATASET_TAGS[$SLURM_ARRAY_TASK_ID]}
field() {
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --dataset-tag "${DATASET_TAG}" \
    --stage "$1" \
    --print-field "$2"
}

RUN_NAME=$(field 1 run_name)
if (( CONTINUE_STAGE == 0 )); then
  CONFIG_VALUE=$(field 1 source_config)
  CHECKPOINT_PATH=$(field 1 source_checkpoint)
  SAMPLE_LABEL=dpm50_source_300k
else
  CONFIG_VALUE=$(field "${CONTINUE_STAGE}" config)
  CHECKPOINT_PATH=$(field "${CONTINUE_STAGE}" expected_checkpoint)
  TARGET_UPDATES=$(field "${CONTINUE_STAGE}" target_total_updates)
  SAMPLE_LABEL=dpm50_cont_$((TARGET_UPDATES / 1000))k
fi

if [[ "${CONFIG_VALUE}" = /* ]]; then
  CONFIG_PATH=${CONFIG_VALUE}
else
  CONFIG_PATH=${PROJECT_DIR}/${CONFIG_VALUE}
fi
OUTPUT_PATH=${SAMPLE_ROOT}/${RUN_NAME}_seed${SEED}_${SAMPLE_LABEL}.npz

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing exact config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Missing exact checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" ]]; then
  echo "Refusing to overwrite existing sample artifact: ${OUTPUT_PATH}" >&2
  exit 1
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

echo "Sampling ${RUN_NAME} at ${SAMPLE_LABEL}"
echo "  checkpoint: ${CHECKPOINT_PATH}"
echo "  config:     ${CONFIG_PATH}"
echo "  output:     ${OUTPUT_PATH}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_PATH}" \
  --num-samples 1 \
  --batch-size 1 \
  --image-size "${IMAGE_SIZE}" \
  --scheduler "${SAMPLER_CLASS}" \
  --num-steps "${SAMPLER_STEPS}" \
  --seed "${SEED}" \
  --class-label "${CLASS_LABEL}" \
  --device cpu \
  --preflight-only

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --scheduler "${SAMPLER_CLASS}" \
  --num-steps "${SAMPLER_STEPS}" \
  --seed "${SEED}" \
  --class-label "${CLASS_LABEL}" \
  --device cuda

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/validate_nf_generalize_fig2_dit_sample.py" \
  --sample "${OUTPUT_PATH}" \
  --sample-root "${SAMPLE_ROOT}" \
  --sample-label "${SAMPLE_LABEL}" \
  --requested-checkpoint "${CHECKPOINT_PATH}" \
  --scheduler "${SAMPLER_CLASS}" \
  --requested-steps "${SAMPLER_STEPS}"
