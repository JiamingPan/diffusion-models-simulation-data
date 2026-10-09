#!/bin/bash
#SBATCH --job-name=n256ctl_samp
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_n256_controls/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_n256_controls/sample_%A_%a.err
#SBATCH -A huterer2
# Held-out samples for the two N = 256 controls (task 0 seed124, task 1 capmatch_tembss), exactly as
# sample_conditioning_mechanism_n256_array.sbatch (63381902): seed 123, 32 held-out cosmologies (CAMELS 900-931)
# x 64 draws, DPM-Solver 50 steps, batch 8, checkpoint-epoch-24999. Writes results/<study>/samples/. Never --overwrite.
# 63381902 tasks took ~8 min with MaxRSS <= 1.5 GB, hence 16 GB.
set -euo pipefail
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
BASE_VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf
PYTHON_BIN=python

cd "${PROJECT_DIR}"
mkdir -p logs/nf_cond_n256_controls
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_cond_capmatch_seed_n256.py --print-train-runs)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
read -r MODEL_RUN STUDY _SEED LEDGER_RUN <<< "${RUNS[$TASK_ID]}"
if [[ -n "${RUN_NAME:-}" && "${RUN_NAME}" != "${LEDGER_RUN}" ]]; then
  echo "RUN_NAME=${RUN_NAME} does not match task ${TASK_ID} (ledger ${LEDGER_RUN})" >&2
  exit 1
fi
MANIFEST="${PROJECT_DIR}/local/${STUDY}/manifest.json"
echo "task ${TASK_ID}: ${MODEL_RUN} (ledger ${LEDGER_RUN}) manifest ${MANIFEST}"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual_n256ctl}}_${TASK_ID}"
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
  --manifest "${MANIFEST}" \
  --run-name "${MODEL_RUN}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 8 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --device cuda
