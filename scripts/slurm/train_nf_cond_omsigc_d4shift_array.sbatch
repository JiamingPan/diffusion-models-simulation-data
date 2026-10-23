#!/bin/bash
#SBATCH --job-name=nf_omscd4train
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --array=0-2%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_d4shift_200k/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_d4shift_200k/train_%A_%a.err
# Ledger nf_cond_omsigc_d4shift. Continuous (Omega_m, sigma_8) cross-attention UNet-128, N = 64, 256, 1024
# (tasks 0, 1, 2 = rows of local/nf_cond_omsigc_d4shift_200k/manifest.json), seed 123, 200k updates.
# Config = baseline local/nf_conditional_omsig_continuous_200k yaml + augmentations
# {RandomDihedral2D p=1, RandomRoll size 128} + new io.output_dir (checked by the preflight receipt).
# Adapted from train_nf_conditional_omsig_continuous_array.sbatch with these differences:
#  * cosmodiff runs from a private, patched copy (same patch list) built per task; the shared
#    cosmo_diffusion_main checkout is not patched in place, and the copy must hash-match the receipt;
#  * the venv is only activated: set_nf_class_conditional_pythonpath is NOT called, because it rewrites
#    00-cosmodiff-base-venv.pth and include-system-site-packages (CLAUDE.md: do not touch); both are
#    checked read-only instead;
#  * exits if the receipt is missing/not PASS or the checkpoint dir exists (cosmodiff_train resumes
#    silently from any checkpoint it finds); --no-requeue; OMP_NUM_THREADS=8 with 8 CPUs.
set -euo pipefail
trap 'rc=$?; echo "[error] training failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/nf_cond_omsigc_d4shift_200k
MANIFEST=${LOCAL}/manifest.json
RECEIPT=${RECEIPT:-${LOCAL}/preflight_receipt.json}
TRAINING_SEED=123
TASK_ID=${SLURM_ARRAY_TASK_ID:?run as an array task}

cd "${PROJECT_DIR}"
[[ -f "${RECEIPT}" ]] || { echo "Missing preflight receipt ${RECEIPT}; run scripts/preflight_nf_cond_omsigc_d4shift.py first" >&2; exit 1; }
grep -q '"status": "PASS"' "${RECEIPT}" || { echo "Preflight receipt ${RECEIPT} is not PASS" >&2; exit 1; }

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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# Run fields from the new manifest, cross-checked against the receipt.
read -r RUN_NAME CONFIG_PATH OUTPUT_DIR < <(python - "${MANIFEST}" "${RECEIPT}" "${TASK_ID}" <<'PY'
import hashlib, json, sys, yaml
from pathlib import Path
manifest, receipt, task = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), int(sys.argv[3])
if hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != receipt["static"]["manifest_sha256"]:
    sys.exit("manifest changed since the preflight receipt")
if not 0 <= task < len(manifest):
    sys.exit(f"array index {task} outside run count {len(manifest)}")
row = manifest[task]
cfg = Path("/home/jiamingp/diffusion_models_repo") / row["config"]
rec = {r["run_name"]: r for r in receipt["static"]["runs"]}[row["run_name"]]
if hashlib.sha256(cfg.read_bytes()).hexdigest() != rec["config_sha256"]:
    sys.exit(f"{cfg} changed since the preflight receipt")
out = yaml.safe_load(cfg.read_text())["io"]["output_dir"]
if out != row["checkpoint_dir"]:
    sys.exit("io.output_dir != manifest checkpoint_dir")
print(row["run_name"], cfg, out)
PY
)
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "Could not read run fields for task ${TASK_ID}" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to train into existing ${OUTPUT_DIR} (cosmodiff_train would resume silently)" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/nf_cond_omsigc_d4shift_runtime/train_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_cond_omsigc_d4shift.py build-runtime --dest "${RUNTIME}" --patch-set train \
  --verify-receipt "${RECEIPT}"
cat >> "${RUNTIME}/stubs/sitecustomize.py" <<'PY'
import os
import random

import numpy as np
import torch

_cosmodiff_seed = int(os.environ["COSMODIFF_TRAIN_SEED"])
random.seed(_cosmodiff_seed)
np.random.seed(_cosmodiff_seed)
torch.manual_seed(_cosmodiff_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_cosmodiff_seed)
PY
COSMODIFF_DIR="${RUNTIME}/cosmodiff_src"
export PYTHONPATH="${RUNTIME}/stubs:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export COSMODIFF_TRAIN_SEED="${TRAINING_SEED}"
export PYTHONHASHSEED="${TRAINING_SEED}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

python -c "import cosmodiff, sys; p = cosmodiff.__file__; print('cosmodiff from', p); sys.exit(0 if p.startswith('${COSMODIFF_DIR}') else 1)"
mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "Training ${RUN_NAME} from a clean initialization -> ${OUTPUT_DIR}"
python scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
