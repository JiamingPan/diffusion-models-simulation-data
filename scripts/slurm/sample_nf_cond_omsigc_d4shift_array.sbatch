#!/bin/bash
#SBATCH --job-name=nf_omscd4samp
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --array=0-2%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_d4shift_200k/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_d4shift_200k/sample_%A_%a.err
# Ledger nf_cond_omsigc_d4shift. Held-out samples (sims 900-931, k = 64, DPMSolverMultistep 50 steps,
# batch 8, seed 123, final checkpoint, no EMA, no guidance), same arguments as
# sample_nf_conditional_omsig_continuous_array.sbatch except: new manifest, NO --overwrite (refuses an
# existing sample file), private patched cosmodiff copy instead of patching the shared checkout, venv
# activated without rewriting the .pth / pyvenv.cfg. Tasks 0, 1, 2 = manifest rows (N = 64, 256, 1024).
set -euo pipefail

PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/nf_cond_omsigc_d4shift_200k
MANIFEST=${LOCAL}/manifest.json
RECEIPT=${RECEIPT:-${LOCAL}/preflight_receipt.json}
TASK_ID=${SLURM_ARRAY_TASK_ID:?run as an array task}

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

read -r RUN_NAME CKPT SAMPLE < <(python - "${MANIFEST}" "${TASK_ID}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/jiamingp/diffusion_models_repo/scripts")
from sample_nf_conditional_bias_probe import output_path_for
rows, task = json.load(open(sys.argv[1])), int(sys.argv[2])
if not 0 <= task < len(rows):
    sys.exit(f"array index {task} outside run count {len(rows)}")
row = rows[task]
k = int(row.get("heldout_samples_per_cosmology", 64))
print(row["run_name"], row["requested_checkpoint"],
      output_path_for(Path("/home/jiamingp/diffusion_models_repo"), row, 123, k, None))
PY
)
[[ -n "${SAMPLE:-}" ]] || { echo "Could not read run fields for task ${TASK_ID}" >&2; exit 1; }
[[ -d "${CKPT}" ]] || { echo "Final checkpoint missing: ${CKPT}" >&2; exit 1; }
[[ ! -e "${SAMPLE}" ]] || { echo "Refusing to overwrite ${SAMPLE}" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/nf_cond_omsigc_d4shift_runtime/sample_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_cond_omsigc_d4shift.py build-runtime --dest "${RUNTIME}" --patch-set sample \
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
