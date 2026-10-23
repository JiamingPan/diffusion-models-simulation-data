#!/bin/bash
#SBATCH --job-name=nf_ftctl300k_train
#SBATCH -A huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_ft_controls_300k/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_ft_controls_300k/train_%A_%a.err
# Ledger nf_cond_omsigc_ft_controls_300k. NO default --array: pass it on the command line (with %2).
# Array task = training segment (manifest train_task); each writes a NEW output dir under
# saved_runs/nf_cond_omsigc_ft_controls_300k/ and runs 1562-1563 epochs x 32 (N = 1024) or 390-391 x 128 (N = 4096):
#   0 A noaug  N1024 seg1  from baseline checkpoint-epoch-6249 -> 7030 (+24,992), 7812 (+50,016)
#   1 A noaug  N4096 seg1  from baseline checkpoint-epoch-1562 -> 1757 (+24,960), 1953 (+50,048)
#   2 B idaug  N1024 seg1  from baseline checkpoint-epoch-6249 -> 7030, 7812
#   3 A noaug  N1024 seg2  from task 0's checkpoint-epoch-7812 -> 9374 (+100,000)     [submit afterok on tasks 0-2]
#   4 A noaug  N4096 seg2  from task 1's checkpoint-epoch-1953 -> 2343 (+99,968)
#   5 B idaug  N1024 seg2  from task 2's checkpoint-epoch-7812 -> 9374
#   6 C d4shift N1024      from aug-arm checkpoint-epoch-7812 (read only) -> 9374
#   7 C d4shift N4096      from aug-arm checkpoint-epoch-1953 (read only) -> 2343
# Optimizer moments, AMP scaler, LR-scheduler state and RNG state are restored from the source checkpoint (no new
# seed; the seed-123 RNG state carries over); EMA restarts (not used at sampling). The wrapper is called with the
# manifest's per-segment opt-in flags (A --allow-no-augmentations; B seg2 and C --allow-checkpoint-augmentations).
# Private cosmodiff copy (stock train patch set) and the wrapper must hash-match the preflight receipt.
set -euo pipefail
trap 'rc=$?; echo "[error] fine-tune failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

[[ "${RUN_NAME:-}" == "nf_cond_omsigc_ft_controls_300k" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=nf_cond_omsigc_ft_controls_300k" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=${PROJECT_DIR}/local/nf_cond_omsigc_ft_controls_300k
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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# Fields: training run, config, source checkpoint, output dir, arm, segment, flags (comma list or "none").
read -r TRAIN_RUN CONFIG_PATH SRC OUTPUT_DIR ARM SEGMENT FLAGS < <(python - "${MANIFEST}" "${RECEIPT}" "${TASK_ID}" <<'PY'
import hashlib, json, sys, yaml
from pathlib import Path
rows, rec, task = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), int(sys.argv[3])
if hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != rec["static"]["manifest_sha256"]:
    sys.exit("manifest changed since the preflight receipt")
grp = [r for r in rows if int(r["train_task"]) == task]
if not grp:
    sys.exit(f"array index {task}: no manifest rows with this train_task")
row = grp[0]
run = row["training_run_name"]
t = rec["static"]["training_runs"][run]
if not rec["resume_check"].get(run):
    sys.exit(f"{run}: no resume check in the preflight receipt")
cfg = Path("/home/jiamingp/diffusion_models_repo") / row["config"]
if hashlib.sha256(cfg.read_bytes()).hexdigest() != t["config_sha256"]:
    sys.exit(f"{cfg} changed since the preflight receipt")
out = yaml.safe_load(cfg.read_text())["io"]["output_dir"]
if out != t["output_dir"] or row["resume_from_checkpoint"] != t["source_checkpoint"] \
        or row["finetune_flags"] != t["finetune_flags"] or int(t["train_task"]) != task:
    sys.exit(f"{run}: output dir / source / flags / task differ from the receipt")
print(run, cfg, row["resume_from_checkpoint"], out, row["arm"], row["segment"],
      ",".join(row["finetune_flags"]) or "none")
PY
)
[[ -n "${FLAGS:-}" ]] || { echo "Could not read run fields for task ${TASK_ID}" >&2; exit 1; }
[[ -d "${SRC}" ]] || { echo "Source checkpoint missing: ${SRC} (seg2: has seg1 finished?)" >&2; exit 1; }
for f in diffusion_pytorch_model.safetensors optimizer.bin scheduler.bin scaler.pt random_states_0.pkl checkpoint_config.yaml; do
  [[ -s "${SRC}/${f}" ]] || { echo "Source ${SRC} lacks ${f}" >&2; exit 1; }
done
if [[ "${SEGMENT}" == "2" && "${ARM}" != "d4shift" ]]; then
  # seg1 must have finished (metrics_finetune.json is written after optim.train returns).
  [[ -s "$(dirname "${SRC}")/metrics_finetune.json" ]] \
    || { echo "seg1 output $(dirname "${SRC}") has no metrics_finetune.json; seg1 not finished" >&2; exit 1; }
fi
case "${ARM}" in
  noaug) [[ ! -e "${SRC}/augmentations.pkl" ]] || { echo "A source ${SRC} has augmentations.pkl" >&2; exit 1; } ;;
  idaug) if [[ "${SEGMENT}" == "1" ]]; then
           [[ ! -e "${SRC}/augmentations.pkl" ]] || { echo "B seg1 source has augmentations.pkl" >&2; exit 1; }
         else
           [[ -s "${SRC}/augmentations.pkl" ]] || { echo "B seg2 source lacks augmentations.pkl" >&2; exit 1; }
         fi ;;
  d4shift) [[ -s "${SRC}/augmentations.pkl" ]] || { echo "C source lacks augmentations.pkl" >&2; exit 1; } ;;
  *) echo "unknown arm ${ARM}" >&2; exit 1 ;;
esac
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to train into existing ${OUTPUT_DIR}" >&2; exit 1; }
FLAG_ARGS=()
[[ "${FLAGS}" == "none" ]] || IFS=',' read -r -a FLAG_ARGS <<< "${FLAGS}"

RUNTIME="${PROJECT_DIR}/results/cache/nf_cond_omsigc_ft_controls_300k_runtime/train_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
[[ ! -e "${RUNTIME}" ]] || { echo "Refusing to reuse runtime dir ${RUNTIME}" >&2; exit 1; }
mkdir -p "$(dirname "${RUNTIME}")"
python scripts/preflight_nf_cond_omsigc_ft_controls_300k.py build-runtime --dest "${RUNTIME}" --patch-set train \
  --verify-receipt "${RECEIPT}"
COSMODIFF_DIR="${RUNTIME}/cosmodiff_src"
export PYTHONPATH="${RUNTIME}/stubs:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "Fine-tuning ${TRAIN_RUN} (arm ${ARM}, segment ${SEGMENT}, flags ${FLAGS}): ${SRC} -> ${OUTPUT_DIR}"
python scripts/finetune_cosmodiff_resume.py --config "${CONFIG_PATH}" --resume-from "${SRC}" ${FLAG_ARGS[@]+"${FLAG_ARGS[@]}"}
