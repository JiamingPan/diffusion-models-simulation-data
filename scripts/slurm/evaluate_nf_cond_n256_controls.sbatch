#!/bin/bash
#SBATCH --job-name=n256ctl_eval
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_n256_controls/evaluate_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_n256_controls/evaluate_%j.err
# CPU evaluation of one N = 256 control, selected by the ledger RUN_NAME (passed with --export):
#   nf_cond_omsigc_n256_seed124        -> local/nf_cond_omsigc_n256_seed124/manifest.json
#   nf_condmech_n256_capmatch_tembss   -> local/nf_condmech_n256_capmatch/manifest.json
#   nf_cond_omsigc_n256_ema_sampling   -> local/nf_cond_omsigc_n256_ema_sampling/manifest.json (2 rows)
# Step 1 mirrors 63381903 (evaluate_nf_conditional_bias_probe.py, frozen VGG probe, seed 123, k 64, bootstrap 2000)
#   -> results/<study>/calibration_vgg
# Step 2 is the ledger metric source: scripts/evaluate_run.py --mode conditional -> results/<study>/evaluate_run_v1
#   (pix_std_ratio_median / copy_fidelity_offset_mean appear only once the pending evaluate_run.py patch is approved).
# Both outputs must not exist; every selected sample file must exist before anything is written.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
case "${RUN_NAME:-}" in
  nf_cond_omsigc_n256_seed124) STUDY=nf_cond_omsigc_n256_seed124 ;;
  nf_condmech_n256_capmatch_tembss) STUDY=nf_condmech_n256_capmatch ;;
  nf_cond_omsigc_n256_ema_sampling) STUDY=nf_cond_omsigc_n256_ema_sampling ;;
  *) echo "RUN_NAME must be one of the three N=256 control ledger runs (got '${RUN_NAME:-}')" >&2; exit 1 ;;
esac
MANIFEST="local/${STUDY}/manifest.json"
CAL_OUT="results/${STUDY}/calibration_vgg"
RUN_OUT="results/${STUDY}/evaluate_run_v1"

cd "${PROJECT_DIR}"
mkdir -p logs/nf_cond_n256_controls results/cache/matplotlib
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest ${MANIFEST}" >&2; exit 1; }
for out in "${CAL_OUT}" "${RUN_OUT}"; do
  [[ ! -e "${out}" ]] || { echo "${out} exists; refusing to overwrite" >&2; exit 1; }
done
source /home/jiamingp/venvs/cosmodiff_nf_class/bin/activate
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
PROBE_RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/n256ctl-eval.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${PROBE_RUNTIME}/sitecustomize.py"
export PYTHONPATH="${PROBE_RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

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

python scripts/evaluate_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/${MANIFEST}" \
  --encoder-type vgg \
  --vgg-encoder results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz \
  --vgg-device cpu \
  --seed 123 \
  --samples-per-cosmology 64 \
  --bootstrap 2000 \
  --output-dir "${PROJECT_DIR}/${CAL_OUT}"

python scripts/evaluate_run.py --mode conditional --manifest "${MANIFEST}" --out-dir "${RUN_OUT}" --device cpu --seed 123 --k 64
