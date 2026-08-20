#!/bin/bash
#SBATCH --job-name=nf_l16_faudit
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh400k/audit_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_fresh400k/audit_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
AUDIT_UPDATES=${AUDIT_UPDATES:-400000}
MANIFEST_PATH=${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_fresh400k/manifest.json

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2_dit_l16_fresh400k
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
"${PYTHON_BIN}" scripts/audit_nf_generalize_fig2_dit_l16_fresh400k_results.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --updates "${AUDIT_UPDATES}"
