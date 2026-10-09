#!/bin/bash
#SBATCH --job-name=evalrun_omsccfg
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --array=0-4%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_cfg_200k/evaluate_run_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_cfg_200k/evaluate_run_%A_%a.err
# Ledger nf_cond_omsigc_cfg. scripts/evaluate_run.py --mode conditional on saved samples, CPU only.
# Task i = manifest row i (N 256, 32768, 1024, 64, 4096); one output dir per N with all nine w:
#   results/nf_cond_omsigc_cfg_200k/evaluate_run_v1/<run>/  (evaluate_run.py creates it; refuses if it exists)
# The manifest is local/nf_cond_omsigc_cfg_200k/eval_manifests/<run>.json: one row per w whose run_name is
# <run>__<guidance_label> and whose sample_path is the concrete guided file, because evaluate_run.py looks up
# output_path_for(..., guidance=None). Extra arguments ("$@") pass through to evaluate_run.py.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
[[ "${RUN_NAME:-}" == "nf_cond_omsigc_cfg" ]] || { echo "Submit with --export=ALL,RUN_NAME=nf_cond_omsigc_cfg" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
LOCAL=local/nf_cond_omsigc_cfg_200k
TASK_ID=${SLURM_ARRAY_TASK_ID:?run as an array task}

cd "${PROJECT_DIR}"
ROW_RUN=$(python3 -c "import json,sys; r=json.load(open('${LOCAL}/manifest.json')); i=int(sys.argv[1]); print(r[i]['run_name'] if 0 <= i < len(r) else '')" "${TASK_ID}")
[[ -n "${ROW_RUN}" ]] || { echo "array index ${TASK_ID} outside manifest" >&2; exit 1; }
MANIFEST=${LOCAL}/eval_manifests/${ROW_RUN}.json
OUT_DIR=results/nf_cond_omsigc_cfg_200k/evaluate_run_v1/${ROW_RUN}
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest ${MANIFEST}" >&2; exit 1; }
[[ ! -e "${OUT_DIR}" ]] || { echo "Refusing to overwrite ${OUT_DIR}" >&2; exit 1; }
mkdir -p results/cache/matplotlib
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1
RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/evalrun-runtime.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${RUNTIME}/sitecustomize.py"
export PYTHONPATH="${RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Every sample file of this N must exist before evaluate_run.py creates OUT_DIR.
python - "${MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
from evaluate_nf_conditional_bias_probe import output_path_for
rows = json.load(open(sys.argv[1]))
root = Path("/home/jiamingp/diffusion_models_repo")
missing = [str(output_path_for(root, r, 123, 64, None)) for r in rows if not output_path_for(root, r, 123, 64, None).exists()]
if missing:
    sys.exit("Missing sample files:\n  " + "\n  ".join(missing))
print(f"{len(rows)} sample files present")
PY

python scripts/evaluate_run.py --mode conditional --manifest "${MANIFEST}" --out-dir "${OUT_DIR}" --device cpu "$@"
