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
EXPECTED_COSMODIFF_COMMIT=${EXPECTED_COSMODIFF_COMMIT:?Set the exact external cosmodiff revision}
CONTINUE_STAGE=${CONTINUE_STAGE:?Set CONTINUE_STAGE to 1, 2, 3, 4, or 5}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-${VENV_PATH}/bin/python}
SWEEP=nf_generalize_fig2_dit_l16_seed_restart500k_v1
PREPARE=${CODE_ROOT}/scripts/prepare_nf_generalize_fig2_dit_l16_seed_restart500k_configs.py
CHECK=${CODE_ROOT}/scripts/check_nf_generalize_fig2_dit_l16_seed_restart500k.py
REPORT_PATH=${PROJECT_DIR}/local/${SWEEP}/preflight_reports/stage${CONTINUE_STAGE}_${SLURM_JOB_ID}.json

if (( CONTINUE_STAGE < 1 || CONTINUE_STAGE > 5 )); then
  echo "CONTINUE_STAGE must be between 1 and 5" >&2
  exit 2
fi
test "$(git -C "${CODE_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${CODE_ROOT}" status --porcelain)"
test "$(git -C "${COSMODIFF_DIR}" rev-parse HEAD)" = "${EXPECTED_COSMODIFF_COMMIT}"
test -z "$(git -C "${COSMODIFF_DIR}" status --porcelain)"
test -x "${PYTHON_BIN}"
test -d "${COSMODIFF_DIR}"
test ! -e "${REPORT_PATH}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/${SWEEP}" "$(dirname "${REPORT_PATH}")"
source "${CODE_ROOT}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

STUB_ROOT=${PROJECT_DIR}/results/cache/python_stubs
mkdir -p "${STUB_ROOT}"
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/write_diffusers_runtime_sitecustomize.py" \
  "${STUB_ROOT}/sitecustomize.py"
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${CODE_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/verify_cosmodiff_seed_restart_runtime.py" \
  "${COSMODIFF_DIR}"

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
    --cosmodiff-dir "${COSMODIFF_DIR}" \
    --expected-class DiTTransformer2DModel
done

echo "PASS: seed-restart stage ${CONTINUE_STAGE} full-state preflight"
