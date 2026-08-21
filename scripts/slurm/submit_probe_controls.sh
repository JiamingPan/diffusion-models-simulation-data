#!/bin/bash
# Login-node gate for the frozen probe-control job. This script never submits
# unless the complete no-GPU preflight succeeds.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CODE_ROOT=${CODE_ROOT:-/scratch/huterer_root/huterer0/jiamingp/probe_controls_code_456f01a}
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
TORCH_HOME=${TORCH_HOME:-/scratch/huterer_root/huterer0/jiamingp/torch_cache}
DATA_ROOT=${DATA_ROOT:-/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG}
INCOMPATIBLE_PYTHON_PATHS=${INCOMPATIBLE_PYTHON_PATHS:-/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
ACCOUNT=${ACCOUNT:-huterer2}
PREFLIGHT_SCRIPT=${PREFLIGHT_SCRIPT:-${CODE_ROOT}/scripts/preflight_probe_controls.sh}
SBATCH_BIN=${SBATCH_BIN:-sbatch}

if [[ -z "${EXPECTED_COMMIT}" ]]; then
  echo "EXPECTED_COMMIT must be set explicitly; refusing an unpinned checkout" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR}" CODE_ROOT="${CODE_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
  TORCH_HOME="${TORCH_HOME}" DATA_ROOT="${DATA_ROOT}" \
  INCOMPATIBLE_PYTHON_PATHS="${INCOMPATIBLE_PYTHON_PATHS}" EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  bash "${PREFLIGHT_SCRIPT}"

mkdir -p "${PROJECT_DIR}/logs/nf_conditional_bias_probe"
cd "${PROJECT_DIR}"
job_id=$(
  PROJECT_DIR="${PROJECT_DIR}" CODE_ROOT="${CODE_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    TORCH_HOME="${TORCH_HOME}" DATA_ROOT="${DATA_ROOT}" \
    INCOMPATIBLE_PYTHON_PATHS="${INCOMPATIBLE_PYTHON_PATHS}" EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    "${SBATCH_BIN}" -A "${ACCOUNT}" --parsable \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CODE_ROOT=${CODE_ROOT},PYTHON_BIN=${PYTHON_BIN},TORCH_HOME=${TORCH_HOME},DATA_ROOT=${DATA_ROOT},INCOMPATIBLE_PYTHON_PATHS=${INCOMPATIBLE_PYTHON_PATHS},EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
    "${CODE_ROOT}/scripts/slurm/run_probe_controls.sbatch"
)
job_id=${job_id%%;*}
echo "probe controls job: ${job_id}"
