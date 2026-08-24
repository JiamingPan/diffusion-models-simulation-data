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
CODE_ROOT=${CODE_ROOT:-${PROJECT_DIR}}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2
REPORT_PATH=${PROJECT_DIR}/local/${SWEEP}/final_audit.json
FINALIZER=${CODE_ROOT}/scripts/finalize_terminal_report.py
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

cd "${PROJECT_DIR}"
mkdir -p "logs/${SWEEP}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${CODE_ROOT}:${PROJECT_DIR}:${PYTHONPATH:-}"
test ! -e "${REPORT_PATH}"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py" \
  --project-dir "${PROJECT_DIR}" \
  --manifest "local/${SWEEP}/manifest.json"

"${PYTHON_BIN}" "${FINALIZER}" finalize "${REPORT_PATH}" \
  --status PASS \
  --job-id "${SLURM_JOB_ID}" \
  --exit-code 0
REPORT_FINALIZED=1
"${PYTHON_BIN}" "${FINALIZER}" require-pass "${REPORT_PATH}" \
  --expected-job-id "${SLURM_JOB_ID}" >/dev/null
echo "PASS: full DiT-L16 continuation artifact audit"
