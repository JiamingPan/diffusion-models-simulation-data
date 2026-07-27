#!/bin/bash
#SBATCH --job-name=nf_l16_fpre
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=00:30:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh300k/precheck_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh300k/precheck_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}
REQUIRE_EMPTY=${REQUIRE_EMPTY:-0}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_fresh300k_configs.py
MANIFEST_PATH=${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_fresh300k/manifest.json

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2_dit_l16_fresh300k
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

"${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --check-only
audit_args=(--manifest "${MANIFEST_PATH}")
if [[ "${REQUIRE_EMPTY}" == "1" ]]; then
  audit_args+=(--require-empty)
fi
"${PYTHON_BIN}" scripts/check_nf_generalize_fig2_dit_l16_fresh300k.py "${audit_args[@]}"

RUN_NAME=$(
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --stage 1 \
    --print-runs | head -1
)
CONFIG_REL=$(
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --stage 1 \
    --run-name "${RUN_NAME}" \
    --print-field config
)
CONFIG_PATH=${PROJECT_DIR}/${CONFIG_REL}

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for DiT precheck')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

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

"${PYTHON_BIN}" scripts/check_nf_generalize_fig2_dit_runtime.py \
  --project-dir "${PROJECT_DIR}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --config "${CONFIG_PATH}" \
  --source-samples-per-file 1 \
  --batch-size 1 \
  --device cuda
echo "Fresh DiT-L16 300k GPU precheck PASS."
