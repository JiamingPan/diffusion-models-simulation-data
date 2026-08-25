#!/bin/bash
# Submit the two seed-restart runs through exact 340k--500k stages.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CODE_ROOT=${CODE_ROOT:?Set CODE_ROOT to the frozen seed-restart code worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the frozen code revision}
EXPECTED_COSMODIFF_BASE_REVISION=${EXPECTED_COSMODIFF_BASE_REVISION:?Set the exact cosmodiff base revision}
COSMODIFF_PIN_ROOT=${COSMODIFF_PIN_ROOT:?Set the immutable cosmodiff pin root}
COSMODIFF_PIN_MANIFEST=${COSMODIFF_PIN_MANIFEST:?Set the immutable cosmodiff pin manifest}
RUNTIME_ROOT=${COSMODIFF_PIN_ROOT}/seed_restart_runtime
ACCOUNT=${ACCOUNT:-huterer2}
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
BASE_VENV_SITE=${BASE_VENV_PATH}/lib/python3.10/site-packages
SYSTEM_ANACONDA_SITE=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/python3.10/site-packages
START_STAGE=${START_STAGE:-1}
REUSE_EXISTING_MANIFEST=${REUSE_EXISTING_MANIFEST:-0}
SWEEP=nf_generalize_fig2_dit_l16_seed_restart500k_v1
PREPARE=${CODE_ROOT}/scripts/prepare_nf_generalize_fig2_dit_l16_seed_restart500k_configs.py

if (( START_STAGE < 1 || START_STAGE > 5 )); then
  echo "START_STAGE must be between 1 and 5" >&2
  exit 2
fi
if (( START_STAGE > 1 )) && [[ "${REUSE_EXISTING_MANIFEST}" != "1" ]]; then
  echo "START_STAGE>1 requires REUSE_EXISTING_MANIFEST=1" >&2
  exit 2
fi
test "$(git -C "${CODE_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${CODE_ROOT}" status --porcelain)"
test -d "${COSMODIFF_PIN_ROOT}"
test -s "${COSMODIFF_PIN_MANIFEST}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME_ROOT}:${CODE_ROOT}:${COSMODIFF_PIN_ROOT}"
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

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/${SWEEP}" \
  "${PROJECT_DIR}/local/${SWEEP}/preflight_reports"

if [[ "${REUSE_EXISTING_MANIFEST}" == "1" ]]; then
  "${PYTHON_BIN}" "${PREPARE}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --upgrade-existing-manifest \
    --check-only
else
  test ! -e "${PROJECT_DIR}/local/${SWEEP}/manifest.json"
  "${PYTHON_BIN}" "${PREPARE}" --project-dir "${PROJECT_DIR}"
fi

"${PYTHON_BIN}" "${PREPARE}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --seed-checkpoints

PREVIOUS_JOB=
for STAGE in $(seq "${START_STAGE}" 5); do
  PRECHECK_ARGS=(
    -A "${ACCOUNT}"
    --parsable
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CODE_ROOT=${CODE_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT},EXPECTED_COSMODIFF_BASE_REVISION=${EXPECTED_COSMODIFF_BASE_REVISION},COSMODIFF_PIN_ROOT=${COSMODIFF_PIN_ROOT},COSMODIFF_PIN_MANIFEST=${COSMODIFF_PIN_MANIFEST},CONTINUE_STAGE=${STAGE}"
  )
  if [[ -n "${PREVIOUS_JOB}" ]]; then
    PRECHECK_ARGS+=(--dependency="afterok:${PREVIOUS_JOB}")
  fi
  PRECHECK_JOB=$(sbatch "${PRECHECK_ARGS[@]}" \
    "${CODE_ROOT}/scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch")
  PRECHECK_JOB=${PRECHECK_JOB%%;*}
  PRECHECK_REPORT=${PROJECT_DIR}/local/${SWEEP}/preflight_reports/stage${STAGE}_${PRECHECK_JOB}.json

  TRAIN_JOB=$(sbatch -A "${ACCOUNT}" \
    --parsable \
    --array=0-1%2 \
    --dependency="afterok:${PRECHECK_JOB}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CODE_ROOT=${CODE_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT},EXPECTED_COSMODIFF_BASE_REVISION=${EXPECTED_COSMODIFF_BASE_REVISION},COSMODIFF_PIN_ROOT=${COSMODIFF_PIN_ROOT},COSMODIFF_PIN_MANIFEST=${COSMODIFF_PIN_MANIFEST},CONTINUE_STAGE=${STAGE},PRECHECK_REPORT=${PRECHECK_REPORT},EXPECTED_PRECHECK_JOB_ID=${PRECHECK_JOB}" \
    "${CODE_ROOT}/scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch")
  TRAIN_JOB=${TRAIN_JOB%%;*}
  echo "stage ${STAGE}: precheck=${PRECHECK_JOB} train=${TRAIN_JOB} report=${PRECHECK_REPORT}"
  PREVIOUS_JOB=${TRAIN_JOB}
done

echo "Submitted two DiT-L16 seed-restart runs (resume seed 456) through 500k."
