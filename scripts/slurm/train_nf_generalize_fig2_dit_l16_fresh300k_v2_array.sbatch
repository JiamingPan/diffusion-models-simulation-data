#!/bin/bash
#SBATCH --job-name=nf_l16_f300
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=48:00:00
#SBATCH --array=0-9%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh300k_v2/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh300k_v2/train_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_fresh300k_v2_configs.py

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2_dit_l16_fresh300k_v2
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

mapfile -t RUNS < <(
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --print-runs
)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

RUN_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
field() {
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --run-name "${RUN_NAME}" \
    --print-field "$1"
}

CONFIG_PATH=${PROJECT_DIR}/$(field config)
CHECKPOINT_DIR=$(field checkpoint_dir)
EXPECTED_CHECKPOINT=$(field expected_checkpoint)

"${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --run-name "${RUN_NAME}" \
  --check-only

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics" "${CHECKPOINT_DIR}"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for DiT training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing cosmodiff training entry point under ${COSMODIFF_DIR}" >&2
  exit 1
fi
COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmodiff branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_dit_patches.lock"
(
  flock -x 9
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_constant_label.py" "${COSMODIFF_DIR}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_dit_class_labels.py" "${COSMODIFF_DIR}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_checkpoint_state.py" "${COSMODIFF_DIR}"
) 9>"${PATCH_LOCK}"

echo "DiT-L16 clean 300k run: ${RUN_NAME}"
echo "  exact target:       ${EXPECTED_CHECKPOINT}"
echo "  walltime:           48 hours"
echo "  recovery interval:  about 5,000 optimizer updates"

"${PYTHON_BIN}" scripts/prune_cosmodiff_checkpoints.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --keep 2 \
  --quarantine-incomplete

"${PYTHON_BIN}" scripts/prune_cosmodiff_checkpoints.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --keep 2 \
  --watch-pid "$$" \
  --interval-seconds 300 &
PRUNER_PID=$!
stop_pruner() {
  kill "${PRUNER_PID}" 2>/dev/null || true
  wait "${PRUNER_PID}" 2>/dev/null || true
}
trap stop_pruner EXIT

if compgen -G "${CHECKPOINT_DIR}/checkpoint-epoch-*" > /dev/null; then
  echo "Recovery checkpoints found; resuming the latest complete checkpoint."
  "${PYTHON_BIN}" scripts/run_cosmodiff_train_with_dit_resume.py \
    --config "${CONFIG_PATH}" \
    --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --target-checkpoint "${EXPECTED_CHECKPOINT}"
else
  echo "No recovery checkpoint found; starting from a clean seed-123 initialization."
  "${PYTHON_BIN}" scripts/run_cosmodiff_train_fresh_seeded.py \
    --config "${CONFIG_PATH}" \
    --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --seed 123
fi

if [[ ! -d "${EXPECTED_CHECKPOINT}" ]]; then
  echo "Training ended without the exact target: ${EXPECTED_CHECKPOINT}" >&2
  echo "Re-submit this array; it will resume from the latest recovery checkpoint." >&2
  exit 1
fi
"${PYTHON_BIN}" scripts/check_nf_generalize_fig2_dit_resume.py \
  --checkpoint "${EXPECTED_CHECKPOINT}" \
  --cosmodiff-dir "${COSMODIFF_DIR}"
stop_pruner
trap - EXIT
"${PYTHON_BIN}" scripts/prune_cosmodiff_checkpoints.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --keep 1
echo "Completed clean 300k run: ${EXPECTED_CHECKPOINT}"
