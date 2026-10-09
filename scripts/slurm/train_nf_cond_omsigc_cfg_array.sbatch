#!/bin/bash
#SBATCH --job-name=nf_omsccfg_train
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=15:00:00
#SBATCH --no-requeue
#SBATCH --array=0-4%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_cfg_200k/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_cfg_200k/train_%A_%a.err
# Ledger nf_cond_omsigc_cfg. Continuous (Omega_m, sigma_8) cross-attention UNet-128 with CFG label dropout 0.1
# and a flagged null [0, 0, 1] (label = [Omega_m_z, sigma_8_z, flag]); seed 123, 200k updates.
# Tasks 0..4 = rows of local/nf_cond_omsigc_cfg_200k/manifest.json = N 256, 32768, 1024, 64, 4096.
# Config = baseline local/nf_conditional_omsig_continuous_200k yaml + io.output_dir, data.label_path,
# encoder_hid_dim 3, cfg_dropout 0.1, cfg_null flag, generate.continuous_labels (checked by the receipt).
# Same structure as train_nf_cond_omsigc_d4shift_array.sbatch: private patched cosmodiff copy (train patch set +
# scripts/patch_cosmodiff_cfg_null_flag.py) that must hash-match the receipt; venv only activated (no .pth /
# pyvenv.cfg rewrite); refuses an existing checkpoint dir (cosmodiff_train would resume silently).
# --mem 32gb: baseline N = 32768 training peaked at 17.6 GB MaxRSS (sacct 62232389_4), so 16 GB is too small.
set -euo pipefail
trap 'rc=$?; echo "[error] training failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

[[ "${RUN_NAME:-}" == "nf_cond_omsigc_cfg" ]] || { echo "Submit with --export=ALL,RUN_NAME=nf_cond_omsigc_cfg" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/nf_cond_omsigc_cfg_200k
MANIFEST=${LOCAL}/manifest.json
RECEIPT=${RECEIPT:-${LOCAL}/preflight_receipt.json}
TRAINING_SEED=123
TASK_ID=${SLURM_ARRAY_TASK_ID:?run as an array task}

cd "${PROJECT_DIR}"
[[ -f "${RECEIPT}" ]] || { echo "Missing preflight receipt ${RECEIPT}" >&2; exit 1; }
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

read -r ROW_RUN CONFIG_PATH OUTPUT_DIR < <(python - "${MANIFEST}" "${RECEIPT}" "${TASK_ID}" <<'PY'
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
c = yaml.safe_load(cfg.read_text())
if c["train"].get("cfg_null") != "flag" or c["model"]["kwargs"].get("encoder_hid_dim") != 3:
    sys.exit("config is not the flagged-null CFG config")
if c["io"]["output_dir"] != row["checkpoint_dir"]:
    sys.exit("io.output_dir != manifest checkpoint_dir")
print(row["run_name"], cfg, c["io"]["output_dir"])
PY
)
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "Could not read run fields for task ${TASK_ID}" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to train into existing ${OUTPUT_DIR} (cosmodiff_train would resume silently)" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/nf_cond_omsigc_cfg_runtime/train_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_cond_omsigc_cfg.py build-runtime --dest "${RUNTIME}" --patch-set train \
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

python -c "import cosmodiff, inspect, sys; from cosmodiff import optim; p = cosmodiff.__file__; print('cosmodiff from', p); sys.exit(0 if p.startswith('${COSMODIFF_DIR}') and 'cfg_null' in inspect.signature(optim.train).parameters else 1)"
mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "Training ${ROW_RUN} from a clean initialization -> ${OUTPUT_DIR}"
python scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
