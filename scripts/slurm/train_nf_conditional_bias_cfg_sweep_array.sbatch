#!/bin/bash
#SBATCH --job-name=nf_bias_cfg_train
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_cfg_sweep/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_cfg_sweep/train_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail
trap 'rc=$?; echo "[error] ${SLURM_JOB_NAME:-nf_bias_cfg_train} failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
BIAS_DATASET_SIZES=${BIAS_DATASET_SIZES:-128,16384}
CFG_DROPOUTS=${CFG_DROPOUTS:-0.05,0.1}
GUIDANCE_SCALES=${GUIDANCE_SCALES:-none,1.0,1.5,2.0}
SAMPLES_PER_COSMOLOGY=${SAMPLES_PER_COSMOLOGY:-64}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_conditional_bias_probe_cfg_sweep results/cache/python_stubs
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "Missing CUDA training env: ${VENV_PATH}" >&2
  exit 1
fi
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_cfg_sweep_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --dataset-sizes "${BIAS_DATASET_SIZES}" \
  --cfg-dropouts "${CFG_DROPOUTS}" \
  --guidance-scales "${GUIDANCE_SCALES}" \
  --sample-k-per-cosmology "${SAMPLES_PER_COSMOLOGY}" \
  --print-train-runs)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#RUNS[@]} )); then
  echo "Array index ${TASK_ID} outside run count ${#RUNS[@]}" >&2
  exit 1
fi
RUN_NAME=${RUNS[$TASK_ID]}
CONFIG_PATH="${PROJECT_DIR}/local/nf_conditional_bias_probe_cfg_sweep/configs/${RUN_NAME}.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Submit prepare first: sbatch -A huterer2 scripts/slurm/prepare_nf_conditional_bias_cfg_sweep.sbatch" >&2
  exit 1
fi

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_JOB_ID:-manual_bias_cfg}_${TASK_ID}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"

export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_bias_cfg_train_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_direct_unet_checkpoint.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_numpy_compat.py" "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT
trap 'rc=$?; echo "[error] ${SLURM_JOB_NAME:-nf_bias_cfg_train} failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

"${PYTHON_BIN}" scripts/check_nf_conditional_bias_probe_runtime.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --require-cuda \
  --allow-cfg

echo "Training CFG sweep ${RUN_NAME}"
echo "config: ${CONFIG_PATH}"
"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
