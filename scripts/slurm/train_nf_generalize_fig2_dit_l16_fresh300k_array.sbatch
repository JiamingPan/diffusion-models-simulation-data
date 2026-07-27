#!/bin/bash
#SBATCH --job-name=nf_l16_fresh
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --array=0-9%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh300k/train_stage%a_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh300k/train_stage%a_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
TRAIN_STAGE=${TRAIN_STAGE:?Set TRAIN_STAGE to a value from 1 through 12}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_fresh300k_configs.py

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2_dit_l16_fresh300k
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

mapfile -t RUNS < <(
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --stage "${TRAIN_STAGE}" \
    --print-runs
)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}; valid range is 0-$(( ${#RUNS[@]} - 1 ))" >&2
  exit 1
fi

RUN_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
field() {
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --stage "${TRAIN_STAGE}" \
    --run-name "${RUN_NAME}" \
    --print-field "$1"
}

CONFIG_REL=$(field config)
CONFIG_PATH=${PROJECT_DIR}/${CONFIG_REL}
CHECKPOINT_DIR=$(field checkpoint_dir)
previous_expected_checkpoint=$(field previous_expected_checkpoint)
expected_checkpoint=$(field expected_checkpoint)

"${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --stage "${TRAIN_STAGE}" \
  --run-name "${RUN_NAME}" \
  --check-only

if [[ -d "${expected_checkpoint}" ]]; then
  echo "Stage ${TRAIN_STAGE} already complete for ${RUN_NAME}: ${expected_checkpoint}"
  exit 0
fi
if [[ "${TRAIN_STAGE}" != "1" && ! -d "${previous_expected_checkpoint}" ]]; then
  echo "Missing exact previous-stage checkpoint: ${previous_expected_checkpoint}" >&2
  exit 1
fi

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
  echo "Missing cosmo_diffusion training entry point under ${COSMODIFF_DIR}" >&2
  exit 1
fi
COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_dit_patches.lock"
(
  flock -x 9
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_constant_label.py" "${COSMODIFF_DIR}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_dit_class_labels.py" "${COSMODIFF_DIR}"
) 9>"${PATCH_LOCK}"

echo "Fresh DiT-L16 ${RUN_NAME}, stage ${TRAIN_STAGE}/12"
echo "  checkpoint directory: ${CHECKPOINT_DIR}"
echo "  exact stage target:    ${expected_checkpoint}"
echo "  walltime:              24 hours"
echo "  recovery interval:     approximately 5,000 optimizer updates"

if compgen -G "${CHECKPOINT_DIR}/checkpoint-epoch-*" > /dev/null; then
  resume_args=(
    --config "${CONFIG_PATH}"
    --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --target-checkpoint "${expected_checkpoint}"
  )
  if [[ "${TRAIN_STAGE}" != "1" ]]; then
    resume_args+=(--minimum-checkpoint "${previous_expected_checkpoint}")
  fi
  "${PYTHON_BIN}" scripts/run_cosmodiff_train_with_dit_resume.py "${resume_args[@]}"
else
  if [[ "${TRAIN_STAGE}" != "1" ]]; then
    echo "No checkpoint exists for noninitial stage ${TRAIN_STAGE}: ${CHECKPOINT_DIR}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" scripts/run_cosmodiff_train_fresh_seeded.py \
    --config "${CONFIG_PATH}" \
    --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --seed 123
fi

if [[ ! -d "${expected_checkpoint}" ]]; then
  echo "Training exited without exact stage target: ${expected_checkpoint}" >&2
  echo "A recovery checkpoint may exist. Re-submit this same stage to resume it." >&2
  exit 1
fi
echo "Completed stage ${TRAIN_STAGE}/12: ${expected_checkpoint}"
