#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
DIFFUSERS_COMPAT_VERSION=${DIFFUSERS_COMPAT_VERSION:-0.31.0}
DIFFUSERS_COMPAT_DIR=${DIFFUSERS_COMPAT_DIR:-${PROJECT_DIR}/local/python_packages/diffusers_${DIFFUSERS_COMPAT_VERSION//./p}}

cd "${PROJECT_DIR}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

mkdir -p "${DIFFUSERS_COMPAT_DIR}"

echo "Installing diffusers==${DIFFUSERS_COMPAT_VERSION} into ${DIFFUSERS_COMPAT_DIR}"
"${PYTHON_BIN}" -m pip install \
  --upgrade \
  --no-deps \
  --target "${DIFFUSERS_COMPAT_DIR}" \
  "diffusers==${DIFFUSERS_COMPAT_VERSION}"

DIFFUSERS_COMPAT_DIR="${DIFFUSERS_COMPAT_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

target = os.environ["DIFFUSERS_COMPAT_DIR"]
sys.path.insert(0, target)

import diffusers
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("diffusers:", diffusers.__version__)
print("diffusers_file:", diffusers.__file__)
if not diffusers.__file__.startswith(target):
    raise SystemExit(f"Did not import diffusers from compatibility target: {target}")
PY

echo
echo "Use this for the class-conditional run:"
echo "  export DIFFUSERS_COMPAT_DIR=${DIFFUSERS_COMPAT_DIR}"
