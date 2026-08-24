#!/bin/bash
#SBATCH --job-name=nf_l16_srpre
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --time=00:45:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_seed_restart500k_v1/precheck_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_seed_restart500k_v1/precheck_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CODE_ROOT=${CODE_ROOT:?Set CODE_ROOT to the frozen seed-restart code worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the frozen code revision}
EXPECTED_COSMODIFF_BASE_REVISION=${EXPECTED_COSMODIFF_BASE_REVISION:?Set the exact cosmodiff base revision}
CONTINUE_STAGE=${CONTINUE_STAGE:?Set CONTINUE_STAGE to 1, 2, 3, 4, or 5}
COSMODIFF_PIN_ROOT=${COSMODIFF_PIN_ROOT:?Set the immutable cosmodiff pin root}
COSMODIFF_PIN_MANIFEST=${COSMODIFF_PIN_MANIFEST:?Set the immutable cosmodiff pin manifest}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-${VENV_PATH}/bin/python}
SWEEP=nf_generalize_fig2_dit_l16_seed_restart500k_v1
PREPARE=${CODE_ROOT}/scripts/prepare_nf_generalize_fig2_dit_l16_seed_restart500k_configs.py
CHECK=${CODE_ROOT}/scripts/check_nf_generalize_fig2_dit_l16_seed_restart500k.py
REPORT_PATH=${PROJECT_DIR}/local/${SWEEP}/preflight_reports/stage${CONTINUE_STAGE}_${SLURM_JOB_ID}.json
FINALIZER=${CODE_ROOT}/scripts/finalize_terminal_report.py
STUB_ROOT=${PROJECT_DIR}/results/cache/python_stubs/seed_restart_precheck_${SLURM_JOB_ID}
REPORT_FINALIZED=0

finalize_on_exit() {
  rc=$?
  trap - EXIT
  if [[ "${REPORT_FINALIZED}" == "0" && -s "${REPORT_PATH}" ]]; then
    set +e
    if (( rc == 0 )); then
      rc=1
    fi
    "${PYTHON_BIN}" "${FINALIZER}" finalize "${REPORT_PATH}" \
      --status FAILED \
      --job-id "${SLURM_JOB_ID}" \
      --exit-code "${rc}"
  fi
  exit "${rc}"
}
trap finalize_on_exit EXIT

if (( CONTINUE_STAGE < 1 || CONTINUE_STAGE > 5 )); then
  echo "CONTINUE_STAGE must be between 1 and 5" >&2
  exit 2
fi
test "$(git -C "${CODE_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${CODE_ROOT}" status --porcelain)"
test -x "${PYTHON_BIN}"
test -d "${COSMODIFF_PIN_ROOT}"
test -s "${COSMODIFF_PIN_MANIFEST}"
test ! -e "${REPORT_PATH}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/${SWEEP}" "$(dirname "${REPORT_PATH}")"
source "${CODE_ROOT}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for DiT seed restart')\n" \
  > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/write_diffusers_runtime_sitecustomize.py" \
  "${STUB_ROOT}/sitecustomize.py"
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_PIN_ROOT}:${CODE_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/verify_cosmodiff_seed_restart_runtime.py" \
  "${COSMODIFF_PIN_ROOT}" \
  --manifest "${COSMODIFF_PIN_MANIFEST}" \
  --expected-base-revision "${EXPECTED_COSMODIFF_BASE_REVISION}" \
  --python-bin "${PYTHON_BIN}" \
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_package_metadata.py" \
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_constant_label.py" \
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_dit_class_labels.py" \
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_checkpoint_state.py"

"${PYTHON_BIN}" "${FINALIZER}" start "${REPORT_PATH}" \
  --job-id "${SLURM_JOB_ID}" >/dev/null

"${PYTHON_BIN}" "${PREPARE}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --check-only

"${PYTHON_BIN}" "${CHECK}" \
  --project-dir "${PROJECT_DIR}" \
  --stage "${CONTINUE_STAGE}" \
  --report "${REPORT_PATH}"

for DATASET_TAG in d2p08 d2p10; do
  CHECKPOINT=$("${PYTHON_BIN}" "${PREPARE}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --dataset-tag "${DATASET_TAG}" \
    --stage "${CONTINUE_STAGE}" \
    --print-field previous_expected_checkpoint)
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/check_nf_generalize_fig2_dit_resume.py" \
    --checkpoint "${CHECKPOINT}" \
    --cosmodiff-dir "${COSMODIFF_PIN_ROOT}" \
    --expected-class DiTTransformer2DModel
done

"${PYTHON_BIN}" "${FINALIZER}" finalize "${REPORT_PATH}" \
  --status PASS \
  --job-id "${SLURM_JOB_ID}" \
  --exit-code 0
REPORT_FINALIZED=1
echo "PASS: seed-restart stage ${CONTINUE_STAGE} full-state preflight"
