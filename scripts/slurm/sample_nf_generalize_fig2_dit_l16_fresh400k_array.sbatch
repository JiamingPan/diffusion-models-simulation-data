#!/bin/bash
#SBATCH --job-name=nf_l16_fsample
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --array=0-9%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh400k/sample_stage%a_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh400k/sample_stage%a_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
SAMPLE_STAGE=${SAMPLE_STAGE:?Set SAMPLE_STAGE to a scientific stage}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_fresh400k_configs.py
SAMPLE_ROOT=${SAMPLE_ROOT:-${PROJECT_DIR}/results/nf_generalize_fig2_dit_l16_fresh400k/samples}
SAMPLER_CLASS=${SAMPLER_CLASS:-DPMSolverMultistepScheduler}
SAMPLER_STEPS=${SAMPLER_STEPS:-50}
NUM_SAMPLES=${NUM_SAMPLES:-512}
BATCH_SIZE=${BATCH_SIZE:-4}
IMAGE_SIZE=${IMAGE_SIZE:-128}
SEED=${SEED:-123}
CLASS_LABEL=${CLASS_LABEL:-0}
OVERWRITE=${OVERWRITE:-0}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2_dit_l16_fresh400k "${SAMPLE_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

mapfile -t RUNS < <(
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --stage "${SAMPLE_STAGE}" \
    --print-runs
)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}; valid range is 0-$(( ${#RUNS[@]} - 1 ))" >&2
  exit 1
fi

RUN_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
field() {
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --stage "${SAMPLE_STAGE}" \
    --run-name "${RUN_NAME}" \
    --print-field "$1"
}

CONFIG_PATH=${PROJECT_DIR}/$(field config)
CHECKPOINT_PATH=$(field expected_checkpoint)
SAMPLE_LABEL=$(field sample_label)
scientific_checkpoint=$(field scientific_checkpoint)
OUTPUT_PATH=${SAMPLE_ROOT}/${RUN_NAME}_seed${SEED}_${SAMPLE_LABEL}.npz

if [[ "${scientific_checkpoint}" != "True" ]]; then
  echo "Stage ${SAMPLE_STAGE} is not a frozen scientific checkpoint." >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Missing exact scientific checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" ]]; then
  echo "Skipping existing exact-checkpoint samples: ${OUTPUT_PATH}"
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

echo "Sampling ${RUN_NAME}, fresh stage ${SAMPLE_STAGE}/16"
echo "  exact checkpoint: ${CHECKPOINT_PATH}"
echo "  sample label:     ${SAMPLE_LABEL}"
echo "  output:           ${OUTPUT_PATH}"

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
