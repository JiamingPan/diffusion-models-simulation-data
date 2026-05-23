#!/bin/bash
#SBATCH --job-name=nf_gen_dpm50
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --array=0-9%4
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_nick_data/sample_dpm50_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_nick_data/sample_dpm50_%A_%a.err
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
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_generalize_nick_data}
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/nf_generalize_nick_data/sampling_configs}
SAMPLE_ROOT=${SAMPLE_ROOT:-${PROJECT_DIR}/results/nf_generalize_nick_data/samples}
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_SAMPLES=${NUM_SAMPLES:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-123}
SAMPLE_LABEL=${SAMPLE_LABEL:-dpm50}
SAMPLER_CLASS=${SAMPLER_CLASS:-DPMSolverMultistepScheduler}
SAMPLER_STEPS=${SAMPLER_STEPS:-50}
OVERWRITE=${OVERWRITE:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_nick_data "${SAMPLE_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

"${PYTHON_BIN}" scripts/prepare_nf_generalize_nick_data_configs.py --project-dir "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/prepare_nf_generalize_nick_data_sampling_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --sample-label "${SAMPLE_LABEL}" \
  --scheduler "${SAMPLER_CLASS}" \
  --num-steps "${SAMPLER_STEPS}" \
  --sample-n "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_generalize_nick_data_configs.py --project-dir "${PROJECT_DIR}" --print-runs)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}; valid range is 0-$(( ${#RUNS[@]} - 1 ))" >&2
  exit 1
fi

RUN_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH=${CONFIG_DIR}/${RUN_NAME}_${SAMPLE_LABEL}.yaml
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints
OUTPUT_PATH=${SAMPLE_ROOT}/${RUN_NAME}_seed${SEED}_${SAMPLE_LABEL}.npz

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff sampling')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

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

echo "Sampling Nick-default ${RUN_NAME}"
echo "  checkpoint root: ${CHECKPOINT_DIR}"
echo "  config:          ${CONFIG_PATH}"
echo "  output:          ${OUTPUT_PATH}"
echo "  samples:         ${NUM_SAMPLES}"
echo "  batch:           ${BATCH_SIZE}"
echo "  sampler:         ${SAMPLE_LABEL} (${SAMPLER_CLASS}, ${SAMPLER_STEPS} steps)"
echo "  cosmodiff:       ${COSMODIFF_DIR}"

"${PYTHON_BIN}" "${COSMODIFF_DIR}/scripts/cosmodiff_sample.py" \
  --config "${CONFIG_PATH}" \
  --filepath "${OUTPUT_PATH}" \
  --n_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --scheduler "${SAMPLER_CLASS}" \
  --num_steps "${SAMPLER_STEPS}" \
  --seed "${SEED}" \
  --device cuda \
  --verbose
