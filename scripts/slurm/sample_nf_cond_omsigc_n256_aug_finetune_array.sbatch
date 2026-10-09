#!/bin/bash
#SBATCH --job-name=nf_omscft_samp
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --array=0-2%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n256_aug_finetune/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n256_aug_finetune/sample_%A_%a.err
# Ledger nf_cond_omsigc_n256_aug_finetune. Tasks 0, 1, 2 = manifest rows = checkpoints at +25k, +50k, +100k updates
# (checkpoint-epoch-28124, 31249, 37499). Held-out sims 900-931, k = 64, DPMSolverMultistep 50 steps, batch 8,
# seed 123, raw weights, no guidance. NO --overwrite. Private cosmodiff copy (stock sample patch set).
set -euo pipefail

[[ "${RUN_NAME:-}" == "nf_cond_omsigc_n256_aug_finetune" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=nf_cond_omsigc_n256_aug_finetune" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/nf_cond_omsigc_n256_aug_finetune
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

read -r ROW_RUN CKPT SAMPLE < <(python - "${MANIFEST}" "${TASK_ID}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/jiamingp/diffusion_models_repo/scripts")
from sample_nf_conditional_bias_probe import output_path_for
rows, task = json.load(open(sys.argv[1])), int(sys.argv[2])
if not 0 <= task < len(rows):
    sys.exit(f"array index {task} outside run count {len(rows)}")
row = rows[task]
print(row["run_name"], row["requested_checkpoint"], output_path_for(Path("/home/jiamingp/diffusion_models_repo"), row, 123, 64, None))
PY
)
[[ -n "${SAMPLE:-}" ]] || { echo "Could not read run fields for task ${TASK_ID}" >&2; exit 1; }
[[ -d "${CKPT}" ]] || { echo "Checkpoint missing: ${CKPT}" >&2; exit 1; }
[[ ! -e "${SAMPLE}" ]] || { echo "Refusing to overwrite ${SAMPLE}" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/nf_cond_omsigc_n256_aug_finetune_runtime/sample_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_cond_omsigc_n256_aug_finetune.py build-runtime --dest "${RUNTIME}" --patch-set sample \
  --verify-receipt "${RECEIPT}"
COSMODIFF_DIR="${RUNTIME}/cosmodiff_src"
export PYTHONPATH="${RUNTIME}/stubs:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

echo "Sampling ${ROW_RUN} from ${CKPT} -> ${SAMPLE}"
python scripts/sample_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --run-name "${ROW_RUN}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 8 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --device cuda
[[ -f "${SAMPLE}" ]] || { echo "Expected sample file not written: ${SAMPLE}" >&2; exit 1; }
