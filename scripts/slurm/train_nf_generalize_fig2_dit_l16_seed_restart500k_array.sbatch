#!/bin/bash
#SBATCH --job-name=nf_l16_sr500
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --time=48:00:00
#SBATCH --array=0-1%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_seed_restart500k_v1/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_seed_restart500k_v1/train_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CODE_ROOT=${CODE_ROOT:?Set CODE_ROOT to the frozen seed-restart code worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the frozen code revision}
EXPECTED_COSMODIFF_BASE_REVISION=${EXPECTED_COSMODIFF_BASE_REVISION:?Set the exact cosmodiff base revision}
CONTINUE_STAGE=${CONTINUE_STAGE:?Set CONTINUE_STAGE to 1, 2, 3, 4, or 5}
PRECHECK_REPORT=${PRECHECK_REPORT:?Set PRECHECK_REPORT to the completed stage report}
EXPECTED_PRECHECK_JOB_ID=${EXPECTED_PRECHECK_JOB_ID:?Set the exact producer precheck job ID}
COSMODIFF_PIN_ROOT=${COSMODIFF_PIN_ROOT:?Set the immutable cosmodiff pin root}
COSMODIFF_PIN_MANIFEST=${COSMODIFF_PIN_MANIFEST:?Set the immutable cosmodiff pin manifest}
RUNTIME_ROOT=${COSMODIFF_PIN_ROOT}/seed_restart_runtime
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-${VENV_PATH}/bin/python}
SWEEP=nf_generalize_fig2_dit_l16_seed_restart500k_v1
PREPARE=${CODE_ROOT}/scripts/prepare_nf_generalize_fig2_dit_l16_seed_restart500k_configs.py
DATASET_TAGS=(d2p08 d2p10)
SYSTEM_ANACONDA_SITE=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/python3.10/site-packages
BASE_VENV_SITE=${BASE_VENV_PATH}/lib/python3.10/site-packages

if (( CONTINUE_STAGE < 1 || CONTINUE_STAGE > 5 )); then
  echo "CONTINUE_STAGE must be between 1 and 5" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 2 )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
test "$(git -C "${CODE_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${CODE_ROOT}" status --porcelain)"
test -s "${PRECHECK_REPORT}"
test -d "${COSMODIFF_PIN_ROOT}"
test -s "${COSMODIFF_PIN_MANIFEST}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/${SWEEP}" \
  "${PROJECT_DIR}/local/${SWEEP}/completions" \
  "${PROJECT_DIR}/results/${SWEEP}/resume_audits"
source "${CODE_ROOT}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

DATASET_TAG=${DATASET_TAGS[$SLURM_ARRAY_TASK_ID]}
field() {
  "${PYTHON_BIN}" "${PREPARE}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --dataset-tag "${DATASET_TAG}" \
    --stage "${CONTINUE_STAGE}" \
    --print-field "$1"
}

RUN_NAME=$(field run_name)
CONFIG_REL=$(field config)
CONFIG_PATH=${PROJECT_DIR}/${CONFIG_REL}
CHECKPOINT_DIR=$(field checkpoint_dir)
MINIMUM_CHECKPOINT=$(field previous_expected_checkpoint)
TARGET_CHECKPOINT=$(field expected_checkpoint)
FINAL_EPOCH=$(field expected_final_epoch)
TARGET_UPDATES=$(field target_total_updates)
STEPS_PER_EPOCH=$(field optimizer_steps_per_epoch)
RESUME_EMA_STEP=$(field previous_expected_ema_step)
TARGET_EMA_STEP=$(field expected_ema_step)
RESUME_SEED=$(field resume_seed)
APPLY_RESUME_SEED=$(field apply_resume_seed)
AUDIT_PATH=${PROJECT_DIR}/results/${SWEEP}/resume_audits/stage${CONTINUE_STAGE}_${DATASET_TAG}_${SLURM_JOB_ID}.json
COMPLETION=${PROJECT_DIR}/local/${SWEEP}/completions/stage${CONTINUE_STAGE}_${DATASET_TAG}_${SLURM_JOB_ID}.json

test -f "${CONFIG_PATH}"
test -d "${CHECKPOINT_DIR}"
test -d "${MINIMUM_CHECKPOINT}"
test ! -e "${AUDIT_PATH}"
test ! -e "${COMPLETION}"

export TORCHDYNAMO_DISABLE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME_ROOT}:${CODE_ROOT}:${COSMODIFF_PIN_ROOT}"
export COSMODIFF_DIR="${COSMODIFF_PIN_ROOT}"
test -s "${RUNTIME_ROOT}/sitecustomize.py"
test -s "${RUNTIME_ROOT}/sklearn/__init__.py"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/verify_cosmodiff_seed_restart_runtime.py" \
  "${COSMODIFF_PIN_ROOT}" \
  --manifest "${COSMODIFF_PIN_MANIFEST}" \
  --expected-base-revision "${EXPECTED_COSMODIFF_BASE_REVISION}" \
  --python-bin "${PYTHON_BIN}" \
  --code-root "${CODE_ROOT}" \
  --expected-torch-prefix "${VENV_PATH}" \
  --incompatible-python-path "${BASE_VENV_SITE}" \
  --incompatible-python-path "${SYSTEM_ANACONDA_SITE}" \
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_package_metadata.py" \
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_constant_label.py" \
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_dit_class_labels.py" \
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_checkpoint_state.py"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/finalize_terminal_report.py" require-pass \
  "${PRECHECK_REPORT}" \
  --expected-job-id "${EXPECTED_PRECHECK_JOB_ID}" >/dev/null

test -f "${COSMODIFF_PIN_ROOT}/scripts/cosmodiff_train.py"
echo "Seed restart ${RUN_NAME}: ${MINIMUM_CHECKPOINT} -> ${TARGET_CHECKPOINT}"
echo "resume_seed=${RESUME_SEED} apply_resume_seed=${APPLY_RESUME_SEED} target_updates=${TARGET_UPDATES}"
echo "resume_ema_step=${RESUME_EMA_STEP} target_ema_step=${TARGET_EMA_STEP}"
START_SECONDS=${SECONDS}

SEED_ARGS=()
if [[ "${APPLY_RESUME_SEED}" == "True" || "${APPLY_RESUME_SEED}" == "true" ]]; then
  SEED_ARGS=(--resume-rng-seed "${RESUME_SEED}")
fi

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/run_cosmodiff_train_with_dit_resume.py" \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_PIN_ROOT}/scripts/cosmodiff_train.py" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --minimum-checkpoint "${MINIMUM_CHECKPOINT}" \
  --target-checkpoint "${TARGET_CHECKPOINT}" \
  "${SEED_ARGS[@]}" \
  --optimizer-steps-per-epoch "${STEPS_PER_EPOCH}" \
  --resume-ema-step "${RESUME_EMA_STEP}" \
  --target-ema-step "${TARGET_EMA_STEP}" \
  --resume-audit "${AUDIT_PATH}" \
  --run-name "${RUN_NAME}" \
  --code-revision "${EXPECTED_COMMIT}"

test -d "${TARGET_CHECKPOINT}"
test -s "${AUDIT_PATH}"
ELAPSED_SECONDS=$((SECONDS - START_SECONDS))
"${PYTHON_BIN}" -c \
  'import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"dataset_tag":sys.argv[2],"continue_stage":int(sys.argv[3]),"source_checkpoint":sys.argv[4],"target_checkpoint":sys.argv[5],"elapsed_seconds":int(sys.argv[6]),"final_epoch":int(sys.argv[7]),"target_total_updates":int(sys.argv[8]),"resume_seed":int(sys.argv[9]),"resume_audit":sys.argv[10],"code_revision":sys.argv[11]},indent=2)+"\n")' \
  "${COMPLETION}" "${DATASET_TAG}" "${CONTINUE_STAGE}" \
  "${MINIMUM_CHECKPOINT}" "${TARGET_CHECKPOINT}" "${ELAPSED_SECONDS}" \
  "${FINAL_EPOCH}" "${TARGET_UPDATES}" "${RESUME_SEED}" "${AUDIT_PATH}" "${EXPECTED_COMMIT}"

echo "PASS: ${DATASET_TAG} stage ${CONTINUE_STAGE} reached ${TARGET_CHECKPOINT}"
