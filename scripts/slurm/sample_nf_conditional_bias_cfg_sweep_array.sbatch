#!/bin/bash
#SBATCH --job-name=nf_bias_cfg_sample
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=08:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_cfg_sweep/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_cfg_sweep/sample_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
BIAS_DATASET_SIZES=${BIAS_DATASET_SIZES:-128,16384}
CFG_DROPOUTS=${CFG_DROPOUTS:-0.05,0.1}
GUIDANCE_SCALES=${GUIDANCE_SCALES:-none,1.0,1.5,2.0}
SAMPLES_PER_COSMOLOGY=${SAMPLES_PER_COSMOLOGY:-64}
SEED=${SEED:-123}
BATCH_SIZE=${BATCH_SIZE:-8}
OVERWRITE=${OVERWRITE:-1}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_conditional_bias_probe_cfg_sweep results/nf_conditional_bias_probe_cfg_sweep/samples results/cache/python_stubs
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
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

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_JOB_ID:-manual_bias_cfg_sample}_${TASK_ID}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff sampling')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"

export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_bias_cfg_sample_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_package_metadata.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_direct_unet_checkpoint.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_numpy_compat.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_sampling_compat.py" "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
IFS=',' read -r -a GUIDANCE_ARRAY <<< "${GUIDANCE_SCALES}"
GUIDANCE_ARGS=()
for scale in "${GUIDANCE_ARRAY[@]}"; do
  GUIDANCE_ARGS+=(--guidance-scale "${scale}")
done

"${PYTHON_BIN}" scripts/sample_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/local/nf_conditional_bias_probe_cfg_sweep/manifest.json" \
  --run-name "${RUN_NAME}" \
  --seed "${SEED}" \
  --samples-per-cosmology "${SAMPLES_PER_COSMOLOGY}" \
  --batch-size "${BATCH_SIZE}" \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --device cuda \
  "${GUIDANCE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
