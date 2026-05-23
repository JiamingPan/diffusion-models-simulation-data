#!/bin/bash
#SBATCH --job-name=nf_cls_cmp
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_class_conditional_u128/sampler_compare_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_class_conditional_u128/sampler_compare_%A_%a.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
INHERITED_COSMODIFF_DIR=${COSMODIFF_DIR:-}
PREFERRED_COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
if [[ -n "${COSMODIFF_DIR_OVERRIDE:-}" ]]; then
  COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE}
else
  COSMODIFF_DIR=${PREFERRED_COSMODIFF_DIR}
fi
if [[ -n "${INHERITED_COSMODIFF_DIR}" && -z "${COSMODIFF_DIR_OVERRIDE:-}" && "${INHERITED_COSMODIFF_DIR}" != "${COSMODIFF_DIR}" ]]; then
  echo "Ignoring inherited COSMODIFF_DIR=${INHERITED_COSMODIFF_DIR}; using ${COSMODIFF_DIR}" >&2
fi
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-123}
NUM_SAMPLES=${NUM_SAMPLES:-96}
FULL_NUM_SAMPLES=${FULL_NUM_SAMPLES:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
OVERWRITE=${OVERWRITE:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
source "${PROJECT_DIR}/scripts/nf_class_conditional_runtime_stubs.sh"
mkdir -p logs/nf_class_conditional_u128 results/nf_class_conditional_u128/sampler_compare
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "Missing class-conditional env: ${VENV_PATH}" >&2
  echo "Run first: bash scripts/setup_nf_class_conditional_env.sh" >&2
  exit 1
fi

set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"
write_nf_class_conditional_python_stubs "${PROJECT_DIR}" sampling
export PYTHONPATH="${NF_CLASS_CONDITIONAL_STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

"${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_sampling_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --comparison-sample-n "${NUM_SAMPLES}" \
  --full-sample-n "${FULL_NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}"

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py --project-dir "${PROJECT_DIR}" --print-runs)
SAMPLER_LABELS=(ddpm500 dpm50)
SAMPLER_CLASSES=(DDPMScheduler DPMSolverMultistepScheduler)
SAMPLER_STEPS=(500 50)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SAMPLER_LABELS[@]} )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}; valid range is 0-$(( ${#SAMPLER_LABELS[@]} - 1 ))" >&2
  exit 1
fi

SAMPLER_LABEL=${SAMPLER_LABELS[$SLURM_ARRAY_TASK_ID]}
SAMPLER_CLASS=${SAMPLER_CLASSES[$SLURM_ARRAY_TASK_ID]}
SAMPLER_STEPS_VALUE=${SAMPLER_STEPS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH="${PROJECT_DIR}/local/nf_class_conditional_u128/sampling_configs/${RUN_NAME}_${SAMPLER_LABEL}_n${NUM_SAMPLES}.yaml"
CHECKPOINT_DIR="/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_class_conditional_u128/${RUN_NAME}_checkpoints"
OUTPUT_PATH="${PROJECT_DIR}/results/nf_class_conditional_u128/sampler_compare/${RUN_NAME}_seed${SEED}_${SAMPLER_LABEL}_n${NUM_SAMPLES}.npz"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing sampler config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_sample.py" ]]; then
  echo "Missing main cosmo_diffusion sampler: ${COSMODIFF_DIR}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_sampling_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_package_metadata.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_sampling_compat.py" "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
COSMODIFF_COMMIT=$(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)
echo "cosmodiff git branch: ${COSMODIFF_BRANCH}"
echo "cosmodiff git commit: ${COSMODIFF_COMMIT}"
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

if [[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" ]]; then
  echo "Skipping existing sample file: ${OUTPUT_PATH}"
  exit 0
fi

echo "Sampling comparison set ${RUN_NAME}"
echo "  checkpoint root: ${CHECKPOINT_DIR}"
echo "  config:          ${CONFIG_PATH}"
echo "  output:          ${OUTPUT_PATH}"
echo "  samples:         ${NUM_SAMPLES}"
echo "  batch:           ${BATCH_SIZE}"
echo "  sampler:         ${SAMPLER_LABEL} (${SAMPLER_CLASS}, ${SAMPLER_STEPS_VALUE} steps)"
echo "  cosmodiff:       ${COSMODIFF_DIR}"
echo "  venv:            ${VENV_PATH}"

"${PYTHON_BIN}" "${COSMODIFF_DIR}/scripts/cosmodiff_sample.py" \
  --config "${CONFIG_PATH}" \
  --filepath "${OUTPUT_PATH}" \
  --n_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --scheduler "${SAMPLER_CLASS}" \
  --num_steps "${SAMPLER_STEPS_VALUE}" \
  --seed "${SEED}" \
  --device cuda \
  --verbose
