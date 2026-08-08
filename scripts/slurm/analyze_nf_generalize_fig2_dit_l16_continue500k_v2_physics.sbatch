#!/bin/bash
#SBATCH --job-name=nf_l16_c5phys
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/physics_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/physics_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2
MANIFEST_PATH=${MANIFEST_PATH:-${PROJECT_DIR}/local/${SWEEP}/analysis_manifest.json}
BASELINE_MANIFEST=${BASELINE_MANIFEST:-${PROJECT_DIR}/local/nf_generalize_fig2_dit/manifest.json}
SAMPLE_ROOT=${SAMPLE_ROOT:-${PROJECT_DIR}/results/${SWEEP}/samples}
TABLE_DIR=${TABLE_DIR:-${PROJECT_DIR}/results/nf_generalize_fig2_dit/tables}
PHYSICS_DIR=${PHYSICS_DIR:-${PROJECT_DIR}/results/nf_generalize_fig2_dit/physics}
SAMPLE_LABEL=${SAMPLE_LABEL:-}
OUT_PREFIX=${OUT_PREFIX:-${SWEEP}}

cd "${PROJECT_DIR}"
mkdir -p "logs/${SWEEP}" "${TABLE_DIR}" "${PHYSICS_DIR}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXTRA_ARGS=()
if [[ -n "${SAMPLE_LABEL}" ]]; then
  EXTRA_ARGS+=(--sample-label "${SAMPLE_LABEL}")
fi

"${PYTHON_BIN}" scripts/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --baseline-manifest "${BASELINE_MANIFEST}" \
  --sample-root "${SAMPLE_ROOT}" \
  --table-dir "${TABLE_DIR}" \
  --physics-dir "${PHYSICS_DIR}" \
  --out-prefix "${OUT_PREFIX}" \
  "${EXTRA_ARGS[@]}"
