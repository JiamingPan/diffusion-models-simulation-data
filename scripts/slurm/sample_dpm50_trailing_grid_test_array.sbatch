#!/bin/bash
#SBATCH --job-name=dpm50_trailing_samp
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
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/dpm50_trailing_grid_test/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/dpm50_trailing_grid_test/sample_%A_%a.err
# Ledger dpm50_trailing_grid_test. Task i = row i of local/dpm50_trailing_grid_test/manifest.json:
#   0: arm a nf_cond_omsigc_hi_u128_d2p15_n32768_fresh200k          (patch set sample)
#   1: arm b nf_cond_omsigc_hi_u128_d2p15_n32768_cfg10flag_fresh200k (patch set sample_cfg, guidance None)
#   2: arm c nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k           (patch set sample)
# Same as the original leading-grid sampling of each arm (final checkpoint, held-out sims 900-931, k = 64,
# DPMSolverMultistep 50 steps, batch 8, seed 123, raw weights, no guidance) except --timestep-spacing trailing.
# Private patched cosmodiff copy (preflight patch set) must hash-match the preflight receipt; the shared
# cosmo_diffusion_main checkout is not touched. Samples -> results/dpm50_trailing_grid_test/samples/*_dpm50_trailing_*.
# NO --overwrite: refuses an existing sample file. Fails unless the sampler logs first_timestep=499 last_timestep=9
# (50 steps) and the npz records the trailing grid [499, ..., 9].
set -euo pipefail

[[ "${RUN_NAME:-}" == "dpm50_trailing_grid_test" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=dpm50_trailing_grid_test" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/dpm50_trailing_grid_test
MANIFEST=${LOCAL}/manifest.json
RECEIPT=${RECEIPT:-${LOCAL}/preflight_receipt.json}
TASK_ID=${SLURM_ARRAY_TASK_ID:?run as an array task}
EXPECTED_LOG="[timestep_spacing] override=trailing spacing=trailing n=50 first_timestep=499 last_timestep=9"

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

read -r ROW_RUN PATCH_SET CKPT SAMPLE < <(python - "${MANIFEST}" "${RECEIPT}" "${TASK_ID}" <<'PY'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "/home/jiamingp/diffusion_models_repo/scripts")
import sample_nf_conditional_bias_probe as w
from sample_nf_conditional_bias_probe import output_path_for
man, rec, task = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3])
if hashlib.sha256(open(man, "rb").read()).hexdigest() != rec["static"]["manifest_sha256"]:
    sys.exit("manifest changed since the preflight receipt")
wp = "/home/jiamingp/diffusion_models_repo/scripts/sample_nf_conditional_bias_probe.py"
if hashlib.sha256(open(wp, "rb").read()).hexdigest() != rec["static"]["wrapper_sha256"]:
    sys.exit("sample_nf_conditional_bias_probe.py changed since the preflight receipt")
if not hasattr(w, "timestep_spacing_cli_args"):
    sys.exit("sample_nf_conditional_bias_probe.py has no --timestep-spacing pass-through")
rows = json.load(open(man))
if not 0 <= task < len(rows):
    sys.exit(f"array index {task} outside run count {len(rows)}")
row = rows[task]
if row.get("ledger_run") != "dpm50_trailing_grid_test" or row.get("timestep_spacing") != "trailing":
    sys.exit("manifest row is not a dpm50_trailing_grid_test trailing row")
path = output_path_for(Path("/home/jiamingp/diffusion_models_repo"), row, 123, 64, None)
if "/results/dpm50_trailing_grid_test/samples/" not in str(path) or "_dpm50_trailing_" not in path.name:
    sys.exit(f"unexpected sample path {path}")
print(row["run_name"], row["runtime_patch_set"], row["requested_checkpoint"], path)
PY
)
[[ -n "${SAMPLE:-}" ]] || { echo "Could not read run fields for task ${TASK_ID}" >&2; exit 1; }
[[ "${PATCH_SET}" == "sample" || "${PATCH_SET}" == "sample_cfg" ]] || { echo "bad patch set ${PATCH_SET}" >&2; exit 1; }
# Training zero-pads epochs below 1000 (N = 32768: checkpoint-epoch-0195); the CFG manifest has the unpadded name.
if [[ ! -d "${CKPT}" ]]; then P="${CKPT%-*}-$(printf '%04d' "$((10#${CKPT##*-}))")"; [[ -d "${P}" ]] && CKPT="${P}"; fi
[[ -d "${CKPT}" ]] || { echo "Final checkpoint missing: ${CKPT}" >&2; exit 1; }
[[ ! -e "${SAMPLE}" ]] || { echo "Refusing to overwrite ${SAMPLE}" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/dpm50_trailing_grid_test_runtime/sample_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
[[ ! -e "${RUNTIME}" ]] || { echo "Refusing to reuse runtime dir ${RUNTIME}" >&2; exit 1; }
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_dpm50_trailing_grid_test.py build-runtime --dest "${RUNTIME}" --patch-set "${PATCH_SET}" \
  --verify-receipt "${RECEIPT}"
COSMODIFF_DIR="${RUNTIME}/cosmodiff_src"
export PYTHONPATH="${RUNTIME}/stubs:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1
TASK_LOG=$(mktemp "${TMPDIR:-/tmp}/dpm50_trailing_${TASK_ID}.XXXXXX.log")

echo "Sampling ${ROW_RUN} (patch set ${PATCH_SET}) from ${CKPT} -> ${SAMPLE}"
python scripts/sample_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --run-name "${ROW_RUN}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 8 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --timestep-spacing trailing \
  --device cuda | tee "${TASK_LOG}"
[[ -f "${SAMPLE}" ]] || { echo "Expected sample file not written: ${SAMPLE}" >&2; exit 1; }
grep -qxF "${EXPECTED_LOG}" "${TASK_LOG}" \
  || { echo "Sampler did not log '${EXPECTED_LOG}'" >&2; exit 1; }
python - "${SAMPLE}" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=True) as z:
    ts = [int(t) for t in z["inference_timesteps"]]
    spacing, ov, shape = str(z["timestep_spacing"]), str(z["timestep_spacing_override"]), z["samples"].shape
want = [499 - 10 * i for i in range(50)]
print(f"npz grid: n={len(ts)} first={ts[0]} last={ts[-1]} spacing={spacing} override={ov} samples={shape}")
if ts != want or spacing != "trailing" or ov != "trailing" or tuple(shape) != (2048, 1, 128, 128):
    sys.exit("sample file does not record the trailing grid [499, ..., 9] with 2048 samples")
PY
echo "OK ${SAMPLE}"
