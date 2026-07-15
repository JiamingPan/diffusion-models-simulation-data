#!/bin/bash
#SBATCH --job-name=nf_l16_pre
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:20:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue/resume_precheck_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue/resume_precheck_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CONTINUE_STAGE=${CONTINUE_STAGE:-1}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2_dit_l16_continue
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

RUN_NAME=$(
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --stage "${CONTINUE_STAGE}" \
    --print-runs | head -1
)
CHECKPOINT=$(
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --stage "${CONTINUE_STAGE}" \
    --run-name "${RUN_NAME}" \
    --print-field previous_expected_checkpoint
)
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Missing exact previous-stage checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for DiT resume precheck')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

PATCH_LOCK="${COSMODIFF_DIR}/.codex_dit_patches.lock"
(
  flock -x 9
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_constant_label.py" "${COSMODIFF_DIR}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_dit_class_labels.py" "${COSMODIFF_DIR}"
) 9>"${PATCH_LOCK}"

echo "Checking DiT continuation checkpoint loader"
echo "  stage:      ${CONTINUE_STAGE}"
echo "  run:        ${RUN_NAME}"
echo "  checkpoint: ${CHECKPOINT}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/check_nf_generalize_fig2_dit_resume.py" \
  --checkpoint "${CHECKPOINT}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --expected-class DiTTransformer2DModel
