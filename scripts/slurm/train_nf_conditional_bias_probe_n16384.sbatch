#!/bin/bash
#SBATCH --job-name=nf_bias_n16k
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/train_n16384_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/train_n16384_%j.err
#SBATCH -A huterer2

set -euo pipefail
trap 'rc=$?; echo "[error] ${SLURM_JOB_NAME:-nf_bias_n16k} failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
BIAS_DATASET_SIZE=16384
BIAS_DATASET_SIZES=${BIAS_DATASET_SIZES:-128,16384}
BIAS_PREPARE_IN_TRAIN=${BIAS_PREPARE_IN_TRAIN:-1}
BIAS_RUN_PREFLIGHT=${BIAS_RUN_PREFLIGHT:-1}
BIAS_PREFLIGHT_ONLY=${BIAS_PREFLIGHT_ONLY:-0}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_bias_probe/prepared_data}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe results/cache/python_stubs
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_JOB_ID:-manual_bias_n16k}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"

export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

if [[ "${BIAS_PREPARE_IN_TRAIN}" == "1" ]]; then
  PREP_LOCK="/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_bias_probe/.prepare.lock"
  mkdir -p "$(dirname "${PREP_LOCK}")"
  while ! mkdir "${PREP_LOCK}" 2>/dev/null; do
    sleep 5
  done
  trap 'rmdir "${PREP_LOCK}" 2>/dev/null || true' EXIT
  "${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_probe_configs.py \
    --project-dir "${PROJECT_DIR}" \
    --prepared-data-root "${PREPARED_DATA_ROOT}" \
    --dataset-sizes "${BIAS_DATASET_SIZES}" \
    --write-arrays \
    --print-table
  rmdir "${PREP_LOCK}" 2>/dev/null || true
  trap - EXIT
else
  echo "Skipping prepare inside train job; expecting dependency-prepared configs/data."
fi

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_probe_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --dataset-sizes "${BIAS_DATASET_SIZE}" \
  --print-runs)
CONFIG_PATH="${PROJECT_DIR}/local/nf_conditional_bias_probe/configs/${RUN_NAME}.yaml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run prepare first: sbatch -A huterer2 scripts/slurm/prepare_nf_conditional_bias_probe.sbatch" >&2
  exit 1
fi
if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing cosmodiff_train.py under ${COSMODIFF_DIR}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_labels_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_direct_unet_checkpoint.py" "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT
trap 'rc=$?; echo "[error] ${SLURM_JOB_NAME:-nf_bias_n16k} failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

if [[ "${BIAS_RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/check_nf_conditional_bias_probe_runtime.py \
    --config "${CONFIG_PATH}" \
    --cosmodiff-dir "${COSMODIFF_DIR}"
fi
if [[ "${BIAS_PREFLIGHT_ONLY}" == "1" ]]; then
  echo "BIAS_PREFLIGHT_ONLY=1; exiting before training."
  exit 0
fi

echo "Training ${RUN_NAME}"
echo "config: ${CONFIG_PATH}"
echo "cosmodiff: ${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
