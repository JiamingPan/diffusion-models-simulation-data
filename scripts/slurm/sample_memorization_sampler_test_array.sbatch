#!/bin/bash
#SBATCH --job-name=memo_sampler
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --array=0-7%4
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/memorization_sampler_test/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/memorization_sampler_test/sample_%A_%a.err
#SBATCH -A huterer2
# Sampler test (sampling only, no training): existing Omega_m-sigma_8 checkpoints at N = 64/256,
# resampled with DDPM 500 (training scheduler) and DPM-Solver 200. Task list: local/memorization_sampler_test/tasks.json.
# Same seed (123), held-out cosmologies and 64 draws as the dpm50 baseline; new output paths, never --overwrite.
set -euo pipefail
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
BASE_VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf
PYTHON_BIN=python

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
read -r TAG RUN_NAME SCHED STEPS < <("${PYTHON_BIN}" -c "import json; t=json.load(open('local/memorization_sampler_test/tasks.json'))[${TASK_ID}]; print(t['sampler_tag'], t['run_name'], t['scheduler'], t['num_steps'])")
echo "task ${TASK_ID}: ${RUN_NAME} ${SCHED} ${STEPS} steps (${TAG})"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export COSMODIFF_DIR TORCHDYNAMO_DISABLE=1

# Same idempotent cosmodiff runtime patches as the original omsig sampling arrays.
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
  --manifest "${PROJECT_DIR}/local/memorization_sampler_test/manifest_${TAG}.json" \
  --run-name "${RUN_NAME}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 32 \
  --scheduler "${SCHED}" \
  --num-steps "${STEPS}" \
  --device cuda
