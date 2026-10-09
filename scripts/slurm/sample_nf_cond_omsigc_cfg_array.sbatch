#!/bin/bash
#SBATCH --job-name=nf_omsccfg_samp
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --array=0-44%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_cfg_200k/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_cfg_200k/sample_%A_%a.err
# Ledger nf_cond_omsigc_cfg. One (N, w) per task from local/nf_cond_omsigc_cfg_200k/sample_tasks.json
# (N-major in training order 256, 32768, 1024, 64, 4096; w in [none, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]).
# Held-out sims 900-931, k = 64, DPMSolverMultistep 50 steps, batch 8, seed 123 (same initial noise for every w),
# final checkpoint, raw weights (no EMA). Guided tasks pass --guidance-scale w; the patched sampler builds the
# null from train.cfg_null = flag, i.e. [0, 0, 1]. NO --overwrite (refuses an existing sample file).
# Private patched cosmodiff copy (sample patch set + cfg_null_flag) must hash-match the receipt.
set -euo pipefail

[[ "${RUN_NAME:-}" == "nf_cond_omsigc_cfg" ]] || { echo "Submit with --export=ALL,RUN_NAME=nf_cond_omsigc_cfg" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/nf_cond_omsigc_cfg_200k
MANIFEST=${LOCAL}/manifest.json
TASKS=${LOCAL}/sample_tasks.json
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

read -r ROW_RUN CKPT GUIDANCE SAMPLE < <(python - "${MANIFEST}" "${TASKS}" "${RECEIPT}" "${TASK_ID}" <<'PY'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "/home/jiamingp/diffusion_models_repo/scripts")
from sample_nf_conditional_bias_probe import output_path_for
man, tasks, rec, task = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), int(sys.argv[4])
for path, key in ((man, "manifest_sha256"), (tasks, "sample_tasks_sha256")):
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != rec["static"][key]:
        sys.exit(f"{path} changed since the preflight receipt")
rows = {r["run_name"]: r for r in json.load(open(man))}
tl = json.load(open(tasks))
if not 0 <= task < len(tl):
    sys.exit(f"array index {task} outside task count {len(tl)}")
t = tl[task]
row = rows[t["run_name"]]
w = t["guidance_scale"]
path = output_path_for(Path("/home/jiamingp/diffusion_models_repo"), row, 123, 64, w)
if str(path) != t["sample_path"]:
    sys.exit("sample path mismatch")
print(row["run_name"], row["requested_checkpoint"], "none" if w is None else repr(float(w)), path)
PY
)
[[ -n "${SAMPLE:-}" ]] || { echo "Could not read task ${TASK_ID}" >&2; exit 1; }
[[ -d "${CKPT}" ]] || { echo "Final checkpoint missing: ${CKPT}" >&2; exit 1; }
[[ ! -e "${SAMPLE}" ]] || { echo "Refusing to overwrite ${SAMPLE}" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/nf_cond_omsigc_cfg_runtime/sample_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_cond_omsigc_cfg.py build-runtime --dest "${RUNTIME}" --patch-set sample \
  --verify-receipt "${RECEIPT}"
COSMODIFF_DIR="${RUNTIME}/cosmodiff_src"
export PYTHONPATH="${RUNTIME}/stubs:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

echo "Sampling ${ROW_RUN} w=${GUIDANCE} from ${CKPT} -> ${SAMPLE}"
python scripts/sample_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --run-name "${ROW_RUN}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 8 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --guidance-scale "${GUIDANCE}" \
  --device cuda
[[ -f "${SAMPLE}" ]] || { echo "Expected sample file not written: ${SAMPLE}" >&2; exit 1; }
