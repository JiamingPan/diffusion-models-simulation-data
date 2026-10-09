#!/bin/bash
#SBATCH --job-name=nf_omscft_train
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n256_aug_finetune/train_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n256_aug_finetune/train_%j.err
# Ledger nf_cond_omsigc_n256_aug_finetune. Continue nf_cond_omsigc_hi_u128_d2p08_n256_fresh200k from
# checkpoint-epoch-24999 (200k updates; read only) for +100k updates (12500 epochs x 8) with
# RandomDihedral2D(p=1) + RandomRoll(size 128), into a NEW output dir; checkpoints every 3125 epochs
# (+25k, +50k, +75k, +100k). Optimizer moments, AMP scaler, LR-scheduler state (CosineAnnealingWarmRestarts T_0 4000)
# and RNG state are restored from the checkpoint; EMA restarts (not used at sampling).
# scripts/finetune_cosmodiff_resume.py keeps the config augmentations (stock train() would replace them with the
# checkpoint's, which are None). Private cosmodiff copy (stock train patch set) must hash-match the receipt.
set -euo pipefail
trap 'rc=$?; echo "[error] fine-tune failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

[[ "${RUN_NAME:-}" == "nf_cond_omsigc_n256_aug_finetune" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=nf_cond_omsigc_n256_aug_finetune" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/nf_cond_omsigc_n256_aug_finetune
MANIFEST=${LOCAL}/manifest.json
RECEIPT=${RECEIPT:-${LOCAL}/preflight_receipt.json}

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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

read -r CONFIG_PATH SRC OUTPUT_DIR < <(python - "${MANIFEST}" "${RECEIPT}" <<'PY'
import hashlib, json, sys, yaml
from pathlib import Path
rows, rec = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
if hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != rec["static"]["manifest_sha256"]:
    sys.exit("manifest changed since the preflight receipt")
cfg = Path("/home/jiamingp/diffusion_models_repo") / rows[0]["config"]
if hashlib.sha256(cfg.read_bytes()).hexdigest() != rec["static"]["config_sha256"]:
    sys.exit(f"{cfg} changed since the preflight receipt")
print(cfg, rows[0]["resume_from_checkpoint"], yaml.safe_load(cfg.read_text())["io"]["output_dir"])
PY
)
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "Could not read run fields" >&2; exit 1; }
[[ -d "${SRC}" ]] || { echo "Source checkpoint missing: ${SRC}" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to train into existing ${OUTPUT_DIR}" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/nf_cond_omsigc_n256_aug_finetune_runtime/train_${SLURM_JOB_ID}"
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_cond_omsigc_n256_aug_finetune.py build-runtime --dest "${RUNTIME}" --patch-set train \
  --verify-receipt "${RECEIPT}"
COSMODIFF_DIR="${RUNTIME}/cosmodiff_src"
export PYTHONPATH="${RUNTIME}/stubs:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "Fine-tuning from ${SRC} -> ${OUTPUT_DIR}"
python scripts/finetune_cosmodiff_resume.py --config "${CONFIG_PATH}" --resume-from "${SRC}"
