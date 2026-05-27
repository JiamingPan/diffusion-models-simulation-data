#!/bin/bash
#SBATCH --job-name=nf_small_ema_sample
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=02:00:00
#SBATCH --array=0-41%4
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_sweep_small_ema/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_sweep_small_ema/sample_%A_%a.err
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
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_sweep_small_ema}
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/nf_sweep_small_ema/configs}
SAMPLE_ROOT=${SAMPLE_ROOT:-${PROJECT_DIR}/results/nf_sweep_small_ema/samples}
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_SAMPLES=${NUM_SAMPLES:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-123}
OVERWRITE=${OVERWRITE:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_sweep_small_ema "${SAMPLE_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_sweep_small_ema_configs.py --project-dir "${PROJECT_DIR}" --print-runs)
EMA_TARGETS=(raw 0.001 0.002 0.003 0.004 0.005 0.006 0.008 0.010 0.015 0.020 0.030 0.040 0.050)
EMA_LABELS=(raw ema0p001 ema0p002 ema0p003 ema0p004 ema0p005 ema0p006 ema0p008 ema0p01 ema0p015 ema0p02 ema0p03 ema0p04 ema0p05)
SAMPLER_LABELS=(train_full dpm25 ddim50)
SAMPLER_SCHEDULERS=("" DPMSolverMultistepScheduler DDIMScheduler)
SAMPLER_STEPS=("" 25 50)

N_EMA=${#EMA_TARGETS[@]}
N_SAMPLER=${#SAMPLER_LABELS[@]}
TASKS_PER_RUN=$((N_EMA * N_SAMPLER))
TOTAL_TASKS=$(( ${#RUNS[@]} * TASKS_PER_RUN ))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}; valid range is 0-$(( TOTAL_TASKS - 1 ))" >&2
  exit 1
fi

RUN_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_RUN))
REM=$((SLURM_ARRAY_TASK_ID % TASKS_PER_RUN))
EMA_INDEX=$((REM / N_SAMPLER))
SAMPLER_INDEX=$((REM % N_SAMPLER))

RUN_NAME=${RUNS[$RUN_INDEX]}
EMA_TARGET=${EMA_TARGETS[$EMA_INDEX]}
EMA_LABEL=${EMA_LABELS[$EMA_INDEX]}
SAMPLER_LABEL=${SAMPLER_LABELS[$SAMPLER_INDEX]}
SAMPLER_CLASS=${SAMPLER_SCHEDULERS[$SAMPLER_INDEX]}
SAMPLER_STEPS_VALUE=${SAMPLER_STEPS[$SAMPLER_INDEX]}

CONFIG_PATH=${CONFIG_DIR}/${RUN_NAME}.yaml
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints
OUTPUT_PATH=${SAMPLE_ROOT}/${RUN_NAME}_seed${SEED}_${EMA_LABEL}_${SAMPLER_LABEL}.npz

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff sampling')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_sampling_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_direct_unet_checkpoint.py" "${COSMODIFF_DIR}"
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

echo "Sampling ${RUN_NAME}"
echo "  checkpoint root: ${CHECKPOINT_DIR}"
echo "  config:          ${CONFIG_PATH}"
echo "  output:          ${OUTPUT_PATH}"
echo "  samples:         ${NUM_SAMPLES}"
echo "  batch:           ${BATCH_SIZE}"
echo "  ema target:      ${EMA_TARGET}"
echo "  sampler:         ${SAMPLER_LABEL}"
echo "  cosmodiff:       ${COSMODIFF_DIR}"

CMD=(
  "${PYTHON_BIN}" "${COSMODIFF_DIR}/scripts/cosmodiff_sample.py"
  --config "${CONFIG_PATH}"
  --filepath "${OUTPUT_PATH}"
  --n_samples "${NUM_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --device cuda
  --verbose
)

if [[ "${EMA_TARGET}" != "raw" ]]; then
  CMD+=(--ema_sigma_rel "${EMA_TARGET}")
fi

if [[ -n "${SAMPLER_CLASS}" ]]; then
  CMD+=(--scheduler "${SAMPLER_CLASS}")
fi

if [[ -n "${SAMPLER_STEPS_VALUE}" ]]; then
  CMD+=(--num_steps "${SAMPLER_STEPS_VALUE}")
fi

"${CMD[@]}"
