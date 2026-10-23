#!/bin/bash
#SBATCH --job-name=evalrun_omscd4
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_d4shift_200k/evaluate_run_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_d4shift_200k/evaluate_run_%j.err
# Ledger nf_cond_omsigc_d4shift. scripts/evaluate_run.py --mode conditional (probe Omega_m median bias,
# bias / W, 68% / 95% coverage, P(k) bands vs real split-half, one-point L1, draw-to-draw and plain
# nearest-training cosine) on saved samples. CPU only. Defaults: the d4shift manifest and
# results/nf_cond_omsigc_d4shift_200k/evaluate_run_v1; MANIFEST / OUT_DIR may be overridden with
# --export (e.g. the no-aug re-score). Extra arguments ("$@", e.g. --run-name) pass through.
# Does not create OUT_DIR (evaluate_run.py does); refuses if it exists or a selected sample file is missing.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
MANIFEST=${MANIFEST:-local/nf_cond_omsigc_d4shift_200k/manifest.json}
OUT_DIR=${OUT_DIR:-results/nf_cond_omsigc_d4shift_200k/evaluate_run_v1}

cd "${PROJECT_DIR}"
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

# Every selected sample file must exist before evaluate_run.py creates OUT_DIR.
python - "${MANIFEST}" "$@" <<'PY'
import argparse, json, sys
from pathlib import Path
from evaluate_nf_conditional_bias_probe import output_path_for
ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--run-name", action="append")
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--k", type=int, default=64)
a, _ = ap.parse_known_args()
rows = json.load(open(a.manifest))
if a.run_name:
    rows = [r for r in rows if r["run_name"] in set(a.run_name)]
if not rows:
    sys.exit("No runs selected.")
root = Path("/home/jiamingp/diffusion_models_repo")
missing = [str(output_path_for(root, r, a.seed, a.k, None)) for r in rows
           if not output_path_for(root, r, a.seed, a.k, None).exists()]
if missing:
    sys.exit("Missing sample files:\n  " + "\n  ".join(missing))
print(f"{len(rows)} sample files present")
PY

python scripts/evaluate_run.py --mode conditional --manifest "${MANIFEST}" --out-dir "${OUT_DIR}" --device cpu "$@"
