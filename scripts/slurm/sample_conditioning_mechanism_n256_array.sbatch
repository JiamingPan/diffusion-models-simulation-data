#!/bin/bash
#SBATCH --job-name=condmech_samp
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditioning_mechanism_n256/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditioning_mechanism_n256/sample_%A_%a.err
#SBATCH -A huterer2
# Held-out samples for the conditioning-mechanism study: same seed, 32 held-out cosmologies x 64 draws,
# DPM-Solver 50 steps and batch size as the baseline omsig continuous sampling. Never --overwrite.
set -euo pipefail
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
BASE_VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf
PYTHON_BIN=python

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditioning_mechanism_n256
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_conditioning_mechanism_study.py --print-train-runs)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
RUN_NAME=${RUNS[$TASK_ID]}
echo "task ${TASK_ID}: ${RUN_NAME}"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual_condmech}}_${TASK_ID}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
printf '\nimport simdiff_eval.timeemb_conditioning as _timeemb\n_timeemb.install()\n' >> "${STUB_ROOT}/sitecustomize.py"
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export COSMODIFF_DIR TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_omsc_sample_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do sleep 1; done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
for patch in package_metadata continuous_labels direct_unet_checkpoint numpy_compat sampling_compat; do
  "${PYTHON_BIN}" "scripts/patch_cosmodiff_${patch}.py" "${COSMODIFF_DIR}"
done
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

"${PYTHON_BIN}" scripts/sample_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/local/nf_conditioning_mechanism_n256/manifest.json" \
  --run-name "${RUN_NAME}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 8 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --device cuda
