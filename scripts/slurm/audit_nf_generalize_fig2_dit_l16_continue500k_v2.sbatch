#!/bin/bash
#SBATCH --job-name=nf_l16_c5audit
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/audit_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/audit_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2

cd "${PROJECT_DIR}"
mkdir -p "logs/${SWEEP}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "local/${SWEEP}/manifest.json"

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

path = Path("local/nf_generalize_fig2_dit_l16_continue500k_v2/final_audit.json")
report = json.loads(path.read_text())
if report.get("analysis_status") != "PASS":
    raise SystemExit(
        f"final analysis audit did not pass: {report.get('analysis_status')}"
    )
print(
    f"{report.get('status')}: DiT-L16 analysis artifacts are valid; "
    f"checkpoint retention={report.get('checkpoint_retention_status')}"
)
PY
