#!/bin/bash
#SBATCH --job-name=nf_omcsamp_n1024
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_om_continuous_200k/sample_n1024_safe_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_om_continuous_200k/sample_n1024_safe_%j.err
# Omega_m-only conditional model, N = 1024 (manifest row 2, nf_cond_omc_hi_u128_d2p10_n1024_fresh200k).
# Held-out samples (sims 900-931, k = 64, DPMSolverMultistep 50 steps, batch 8, seed 123, final checkpoint
# epoch 6249, no EMA, no guidance): the same sample_nf_conditional_bias_probe.py arguments and manifest as
# sample_nf_conditional_om_continuous_array.sbatch (N = 64, 256: job 63397804), except
#   * no nf_class_conditional_pythonpath.sh: pyvenv.cfg and 00-cosmodiff-base-venv.pth are only checked,
#     never written;
#   * no patch_cosmodiff_*.py against the shared checkout: the same five patches are applied to a private
#     copy, whose file hashes must equal preflight_n1024_safe_receipt.json;
#   * single run, not an array; refuses an existing sample file; no --overwrite.
set -euo pipefail

PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
MANIFEST=${PROJECT_DIR}/local/nf_conditional_om_continuous_200k/manifest.json
RECEIPT=${PROJECT_DIR}/local/nf_conditional_om_continuous_200k/preflight_n1024_safe_receipt.json
EXPECTED_RUN=nf_cond_omc_hi_u128_d2p10_n1024_fresh200k
ROW_INDEX=2

cd "${PROJECT_DIR}"
[[ -f "${RECEIPT}" ]] && grep -q '"status": "PASS"' "${RECEIPT}" \
  || { echo "Missing or non-PASS preflight receipt ${RECEIPT}" >&2; exit 1; }
source "${VENV_PATH}/bin/activate"
grep -qx 'include-system-site-packages = true' "${VENV_PATH}/pyvenv.cfg" \
  || { echo "venv pyvenv.cfg changed; not touching it" >&2; exit 1; }
[[ -s "${VENV_PATH}/lib/python3.10/site-packages/00-cosmodiff-base-venv.pth" ]] \
  || { echo "00-cosmodiff-base-venv.pth missing; not touching it" >&2; exit 1; }
FILTERED=""
IFS=':' read -r -a _pp <<< "${PYTHONPATH:-}"
for e in "${_pp[@]}"; do
  [[ -z "${e}" || "${e}" == *site-packages* || "${e}" == *dist-packages* ]] && continue
  FILTERED="${FILTERED:+${FILTERED}:}${e}"
done
export PYTHONPATH="${FILTERED}"

read -r RUN_NAME CKPT SAMPLE < <(python - "${MANIFEST}" "${ROW_INDEX}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/jiamingp/diffusion_models_repo/scripts")
from sample_nf_conditional_bias_probe import output_path_for
rows, i = json.load(open(sys.argv[1])), int(sys.argv[2])
row = rows[i]
k = int(row.get("heldout_samples_per_cosmology", 64))
print(row["run_name"], row["requested_checkpoint"],
      output_path_for(Path("/home/jiamingp/diffusion_models_repo"), row, 123, k, None))
PY
)
[[ "${RUN_NAME:-}" == "${EXPECTED_RUN}" ]] || { echo "Manifest row ${ROW_INDEX} is '${RUN_NAME:-}', expected ${EXPECTED_RUN}" >&2; exit 1; }
[[ -d "${CKPT}" ]] || { echo "Final checkpoint missing: ${CKPT}" >&2; exit 1; }
[[ ! -e "${SAMPLE}" ]] || { echo "Refusing to overwrite ${SAMPLE}" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/nf_conditional_om_continuous_n1024_runtime/sample_${SLURM_JOB_ID:?run under slurm}"
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_conditional_om_continuous_n1024.py build-runtime --dest "${RUNTIME}" \
  --verify-receipt "${RECEIPT}"
COSMODIFF_DIR="${RUNTIME}/cosmodiff_src"
export PYTHONPATH="${RUNTIME}/stubs:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

echo "Sampling ${RUN_NAME} from ${CKPT} -> ${SAMPLE}"
python scripts/sample_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --run-name "${RUN_NAME}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 8 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --device cuda
[[ -f "${SAMPLE}" ]] || { echo "Expected sample file not written: ${SAMPLE}" >&2; exit 1; }
