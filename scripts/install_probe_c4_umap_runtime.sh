#!/bin/bash
# Install the exact C4 UMAP stack into an isolated folder.

set -euo pipefail
trap 'rc=$?; echo "[c4-umap-install] failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

CODE_ROOT=${CODE_ROOT:?Set CODE_ROOT to the frozen code worktree}
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
INCOMPATIBLE_PYTHON_PATHS=${INCOMPATIBLE_PYTHON_PATHS:-/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages}
UMAP_SITE_PACKAGES=${UMAP_SITE_PACKAGES:-/scratch/huterer_root/huterer0/jiamingp/probe_c4_umap_py_0p5p5_0p5p10}

test -x "${PYTHON_BIN}"
test -f "${CODE_ROOT}/scripts/probe_controls_runtime.sh"

export PYTHONNOUSERSITE=1
source "${CODE_ROOT}/scripts/probe_controls_runtime.sh"
probe_controls_prepare_runtime "${CODE_ROOT}" "${PYTHON_BIN}" "${INCOMPATIBLE_PYTHON_PATHS}"

validate_runtime() {
  PYTHONPATH="${UMAP_SITE_PACKAGES}:${PYTHONPATH}" \
  UMAP_SITE_PACKAGES="${UMAP_SITE_PACKAGES}" \
  "${PYTHON_BIN}" -c '
import os
from pathlib import Path
import llvmlite
import numba
import numpy
import pynndescent
import umap

runtime = Path(os.environ["UMAP_SITE_PACKAGES"]).resolve()
assert numpy.__version__ == "1.26.4", numpy.__version__
assert umap.__version__ == "0.5.5", umap.__version__
assert pynndescent.__version__ == "0.5.10", pynndescent.__version__
assert numba.__version__ == "0.59.1", numba.__version__
assert llvmlite.__version__ == "0.42.0", llvmlite.__version__
assert Path(umap.__file__).resolve().is_relative_to(runtime), umap.__file__
assert Path(pynndescent.__file__).resolve().is_relative_to(runtime), pynndescent.__file__
assert Path(numba.__file__).resolve().is_relative_to(runtime), numba.__file__
assert Path(llvmlite.__file__).resolve().is_relative_to(runtime), llvmlite.__file__
print("[c4-umap-install] validated", runtime)
'
}

if [[ -d "${UMAP_SITE_PACKAGES}" ]]; then
  validate_runtime
  echo "[c4-umap-install] exact isolated runtime already exists"
  exit 0
fi
if [[ -e "${UMAP_SITE_PACKAGES}" ]]; then
  echo "ERROR: runtime target exists but is not a directory: ${UMAP_SITE_PACKAGES}" >&2
  exit 1
fi

RUNTIME_PARENT=$(dirname "${UMAP_SITE_PACKAGES}")
RUNTIME_NAME=$(basename "${UMAP_SITE_PACKAGES}")
mkdir -p "${RUNTIME_PARENT}"
TEMP_DIR=$(mktemp -d "${RUNTIME_PARENT}/.${RUNTIME_NAME}.tmp.XXXXXX")
cleanup() {
  if [[ -n "${TEMP_DIR:-}" && -d "${TEMP_DIR}" ]]; then
    rm -rf "${TEMP_DIR}"
  fi
}
trap cleanup EXIT

"${PYTHON_BIN}" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --target "${TEMP_DIR}" \
  "umap-learn==0.5.5" \
  "pynndescent==0.5.10" \
  "numba==0.59.1" \
  "llvmlite==0.42.0"

UMAP_SITE_PACKAGES="${TEMP_DIR}" validate_runtime
mv "${TEMP_DIR}" "${UMAP_SITE_PACKAGES}"
TEMP_DIR=
validate_runtime
echo "[c4-umap-install] installed exact isolated runtime at ${UMAP_SITE_PACKAGES}"
