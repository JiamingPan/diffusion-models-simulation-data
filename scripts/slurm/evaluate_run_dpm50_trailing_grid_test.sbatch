#!/bin/bash
#SBATCH --job-name=evalrun_dpm50_trailing
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/dpm50_trailing_grid_test/evaluate_run_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/dpm50_trailing_grid_test/evaluate_run_%j.err
# CPU only. Ledger dpm50_trailing_grid_test: scripts/evaluate_run.py --mode conditional on the three trailing-grid
# sample files (results/dpm50_trailing_grid_test/samples/*_dpm50_trailing_*.npz, read only), same flags as the
# leading-grid references (evaluate_run_v1 of nf_cond_omsigc_cfg, evaluate_run_plus0, and the arm-a re-score)
# -> NEW dir results/dpm50_trailing_grid_test/evaluate_run_trailing (evaluate_run.py refuses an existing --out-dir).
# Refuses to start unless every sample file exists and records the trailing grid [499, ..., 9].
# Submit with --dependency=afterok:<sample array job id>. Extra arguments ("$@") pass through to evaluate_run.py.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
[[ "${RUN_NAME:-}" == "dpm50_trailing_grid_test" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=dpm50_trailing_grid_test" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
MANIFEST=local/dpm50_trailing_grid_test/manifest.json
OUT_DIR=results/dpm50_trailing_grid_test/evaluate_run_trailing
RUNS=(nf_cond_omsigc_hi_u128_d2p15_n32768_fresh200k nf_cond_omsigc_hi_u128_d2p15_n32768_cfg10flag_fresh200k
      nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k)

cd "${PROJECT_DIR}"
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest ${MANIFEST}" >&2; exit 1; }
[[ ! -e "${OUT_DIR}" ]] || { echo "Refusing to overwrite ${OUT_DIR}" >&2; exit 1; }
mkdir -p results/cache/matplotlib "$(dirname "${OUT_DIR}")"
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1
RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/evalrun-runtime.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${RUNTIME}/sitecustomize.py"
export PYTHONPATH="${RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - "${MANIFEST}" "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from evaluate_nf_conditional_bias_probe import output_path_for
rows = {r["run_name"]: r for r in json.load(open(sys.argv[1]))}
root = Path("/home/jiamingp/diffusion_models_repo")
want = [499 - 10 * i for i in range(50)]
bad = []
for n in sys.argv[2:]:
    p = output_path_for(root, rows[n], 123, 64, None)
    if "_dpm50_trailing_" not in p.name or not p.exists():
        bad.append(f"missing or misnamed: {p}")
        continue
    with np.load(p, allow_pickle=True) as z:
        ts = [int(t) for t in z["inference_timesteps"]] if "inference_timesteps" in z.files else []
        sp = str(z["timestep_spacing"]) if "timestep_spacing" in z.files else ""
    if ts != want or sp != "trailing":
        bad.append(f"not a trailing-grid file: {p}")
    print(f"{p}: n={len(ts)} first={ts[0] if ts else None} last={ts[-1] if ts else None} spacing={sp}")
if bad:
    sys.exit("\n  ".join(["Sample check failed:"] + bad))
print(f"{len(sys.argv) - 2} trailing-grid sample files present")
PY

ARGS=()
for r in "${RUNS[@]}"; do ARGS+=(--run-name "${r}"); done
python scripts/evaluate_run.py --mode conditional --manifest "${MANIFEST}" "${ARGS[@]}" --out-dir "${OUT_DIR}" --device cpu "$@"
