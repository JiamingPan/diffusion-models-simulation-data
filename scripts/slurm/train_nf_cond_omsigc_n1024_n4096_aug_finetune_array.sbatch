#!/bin/bash
#SBATCH --job-name=nf_omscft_n1k4k_train
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --array=0-1%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n1024_n4096_aug_finetune/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n1024_n4096_aug_finetune/train_%A_%a.err
# Ledger nf_cond_omsigc_n1024_n4096_aug_finetune. Array task = training run in manifest order:
#   0: nf_cond_omsigc_hi_u128_d2p10_n1024_fresh200k from checkpoint-epoch-6249 (read only), 1563 epochs x 32
#      = +50,016 updates, checkpoint_every_n_epochs 7031 -> checkpoint-epoch-7030 (+24,992) and 7812 (+50,016)
#   1: nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k from checkpoint-epoch-1562 (read only), 391 epochs x 128
#      = +50,048 updates, checkpoint_every_n_epochs 1758 -> checkpoint-epoch-1757 (+24,960) and 1953 (+50,048)
# RandomDihedral2D(p=1) + RandomRoll(size 128), labels untouched, seed-123 baselines (RNG state restored, see below),
# into NEW output dirs under saved_runs/nf_cond_omsigc_n1024_n4096_aug_finetune/. Optimizer moments, AMP scaler,
# LR-scheduler state (CosineAnnealingWarmRestarts T_0 4000) and RNG state are restored from the checkpoint; EMA
# restarts (not used at sampling). scripts/finetune_cosmodiff_resume.py keeps the config augmentations (stock
# train() would replace them with the checkpoint's, which are None) and refuses an existing output dir.
# Private cosmodiff copy (stock train patch set) must hash-match the preflight receipt.
set -euo pipefail
trap 'rc=$?; echo "[error] fine-tune failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

[[ "${RUN_NAME:-}" == "nf_cond_omsigc_n1024_n4096_aug_finetune" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=nf_cond_omsigc_n1024_n4096_aug_finetune" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/nf_cond_omsigc_n1024_n4096_aug_finetune
MANIFEST=${LOCAL}/manifest.json
RECEIPT=${RECEIPT:-${LOCAL}/preflight_receipt.json}
TASK_ID=${SLURM_ARRAY_TASK_ID:?run as an array task}
# Seed: as in the N = 256 fine-tune, no new seed is set; the RNG state of the seed-123 baseline is restored from the
# checkpoint by accelerator.load_state (COSMODIFF_TRAIN_SEED / PYTHONHASHSEED are not exported here either).

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

read -r TRAIN_RUN CONFIG_PATH SRC OUTPUT_DIR < <(python - "${MANIFEST}" "${RECEIPT}" "${TASK_ID}" <<'PY'
import hashlib, json, sys, yaml
from pathlib import Path
rows, rec, task = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), int(sys.argv[3])
if hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != rec["static"]["manifest_sha256"]:
    sys.exit("manifest changed since the preflight receipt")
runs = []
for r in rows:
    if r["training_run_name"] not in runs:
        runs.append(r["training_run_name"])
if not 0 <= task < len(runs):
    sys.exit(f"array index {task} outside training-run count {len(runs)}")
run = runs[task]
row = next(r for r in rows if r["training_run_name"] == run)
t = rec["static"]["training_runs"][run]
if run not in rec["resume_check"] or not rec["resume_check"][run]:
    sys.exit(f"{run}: no resume check in the preflight receipt")
cfg = Path("/home/jiamingp/diffusion_models_repo") / row["config"]
if hashlib.sha256(cfg.read_bytes()).hexdigest() != t["config_sha256"]:
    sys.exit(f"{cfg} changed since the preflight receipt")
out = yaml.safe_load(cfg.read_text())["io"]["output_dir"]
if out != t["output_dir"] or row["resume_from_checkpoint"] != t["source_checkpoint"]:
    sys.exit(f"{run}: output dir / source differ from the receipt")
print(run, cfg, row["resume_from_checkpoint"], out)
PY
)
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "Could not read run fields for task ${TASK_ID}" >&2; exit 1; }
[[ -d "${SRC}" ]] || { echo "Source checkpoint missing: ${SRC}" >&2; exit 1; }
[[ ! -e "${SRC}/augmentations.pkl" ]] || { echo "Source ${SRC} has augmentations.pkl" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to train into existing ${OUTPUT_DIR}" >&2; exit 1; }

RUNTIME="${PROJECT_DIR}/results/cache/nf_cond_omsigc_n1024_n4096_aug_finetune_runtime/train_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
[[ ! -e "${RUNTIME}" ]] || { echo "Refusing to reuse runtime dir ${RUNTIME}" >&2; exit 1; }
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_cond_omsigc_n1024_n4096_aug_finetune.py build-runtime --dest "${RUNTIME}" --patch-set train \
  --verify-receipt "${RECEIPT}"
COSMODIFF_DIR="${RUNTIME}/cosmodiff_src"
export PYTHONPATH="${RUNTIME}/stubs:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "Fine-tuning ${TRAIN_RUN}: ${SRC} -> ${OUTPUT_DIR}"
python scripts/finetune_cosmodiff_resume.py --config "${CONFIG_PATH}" --resume-from "${SRC}"
