#!/bin/bash
#SBATCH --job-name=nf_l16_c500
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --time=48:00:00
#SBATCH --array=0-9%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/train_stage%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/train_stage%A_%a.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CONTINUE_STAGE=${CONTINUE_STAGE:?Set CONTINUE_STAGE to 1, 2, 3, 4, or 5}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2
PREPARE=${PROJECT_DIR}/scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py
PRECHECK_REPORT=${PROJECT_DIR}/local/${SWEEP}/precheck_report.json
DATASET_TAGS=(d2p06 d2p07 d2p08 d2p09 d2p10 d2p11 d2p12 d2p13 d2p14 d2p15)

if (( CONTINUE_STAGE < 1 || CONTINUE_STAGE > 5 )); then
  echo "CONTINUE_STAGE must be between 1 and 5" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#DATASET_TAGS[@]} )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/${SWEEP}" \
  "${PROJECT_DIR}/local/${SWEEP}/completions"
test -s "${PRECHECK_REPORT}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

DATASET_TAG=${DATASET_TAGS[$SLURM_ARRAY_TASK_ID]}
field() {
  "${PYTHON_BIN}" "${PREPARE}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --dataset-tag "${DATASET_TAG}" \
    --stage "${CONTINUE_STAGE}" \
    --print-field "$1"
}

"${PYTHON_BIN}" "${PREPARE}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --dataset-tag "${DATASET_TAG}" \
  --stage "${CONTINUE_STAGE}" \
  --check-only

RUN_NAME=$(field run_name)
CONFIG_REL=$(field config)
CONFIG_PATH=${PROJECT_DIR}/${CONFIG_REL}
CHECKPOINT_DIR=$(field checkpoint_dir)
MINIMUM_CHECKPOINT=$(field previous_expected_checkpoint)
TARGET_CHECKPOINT=$(field expected_checkpoint)
FINAL_EPOCH=$(field expected_final_epoch)
TARGET_UPDATES=$(field target_total_updates)
CONFIG_DIGEST=$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')
COMPLETION=${PROJECT_DIR}/local/${SWEEP}/completions/stage${CONTINUE_STAGE}_${DATASET_TAG}.json

test -f "${CONFIG_PATH}"
test -d "${CHECKPOINT_DIR}"
test -d "${MINIMUM_CHECKPOINT}"

STUB_ROOT=${PROJECT_DIR}/results/cache/python_stubs
mkdir -p "${STUB_ROOT}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/write_diffusers_runtime_sitecustomize.py" \
  "${STUB_ROOT}/sitecustomize.py"
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

test -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

PATCH_LOCK=${COSMODIFF_DIR}/.codex_dit_patches.lock
(
  flock -x 9
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_constant_label.py" "${COSMODIFF_DIR}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_dit_class_labels.py" "${COSMODIFF_DIR}"
) 9>"${PATCH_LOCK}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/check_nf_generalize_fig2_dit_resume.py" \
  --checkpoint "${MINIMUM_CHECKPOINT}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --expected-class DiTTransformer2DModel

echo "Continuing ${RUN_NAME}: ${MINIMUM_CHECKPOINT} -> ${TARGET_CHECKPOINT}"
echo "Target updates: ${TARGET_UPDATES}; recovery checkpoints: about every 5k updates"
START_SECONDS=${SECONDS}

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/run_cosmodiff_train_with_dit_resume.py" \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --minimum-checkpoint "${MINIMUM_CHECKPOINT}" \
  --target-checkpoint "${TARGET_CHECKPOINT}"

test -d "${TARGET_CHECKPOINT}"
ELAPSED_SECONDS=$((SECONDS - START_SECONDS))
"${PYTHON_BIN}" -c \
  'import json, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"dataset_tag": sys.argv[2], "continue_stage": int(sys.argv[3]), "source_checkpoint": sys.argv[4], "target_checkpoint": sys.argv[5], "elapsed_seconds": int(sys.argv[6]), "final_epoch": int(sys.argv[7]), "target_total_updates": int(sys.argv[8]), "config_sha256": sys.argv[9]}, indent=2) + "\n")' \
  "${COMPLETION}" "${DATASET_TAG}" "${CONTINUE_STAGE}" \
  "${MINIMUM_CHECKPOINT}" "${TARGET_CHECKPOINT}" "${ELAPSED_SECONDS}" \
  "${FINAL_EPOCH}" "${TARGET_UPDATES}" "${CONFIG_DIGEST}"

test -s "${COMPLETION}"
echo "PASS: ${DATASET_TAG} stage ${CONTINUE_STAGE} reached ${TARGET_CHECKPOINT}"
