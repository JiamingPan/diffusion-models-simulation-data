#!/bin/bash
#SBATCH --job-name=nf_omceval_n1024
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_om_continuous_200k/evaluate_n1024_safe_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_om_continuous_200k/evaluate_n1024_safe_%j.err
# Omega_m-only conditional model, N = 1024: VGG bias-probe calibration of the seed-123 k=64 held-out samples.
# Same evaluate_nf_conditional_bias_probe.py arguments and manifest as
# evaluate_nf_conditional_om_continuous.sbatch (used for calibration_vgg_n64_n256), except
#   * --run-name fixed to nf_cond_omc_hi_u128_d2p10_n1024_fresh200k; OUTPUT_DIR fixed to calibration_vgg_n1024;
#   * refuses to start if OUTPUT_DIR exists (evaluate_nf_conditional_bias_probe.py itself writes into an
#     existing dir with mkdir exist_ok=True) or the sample file is missing; does not pre-create OUTPUT_DIR;
#   * checks (read-only) that the venv .pth / pyvenv.cfg are present; never writes them. CPU only.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
MANIFEST=${PROJECT_DIR}/local/nf_conditional_om_continuous_200k/manifest.json
RUN=nf_cond_omc_hi_u128_d2p10_n1024_fresh200k
VGG_ENCODER=results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz
OUTPUT_DIR=results/nf_conditional_om_continuous_200k/calibration_vgg_n1024
SAMPLE=${PROJECT_DIR}/results/nf_conditional_om_continuous_200k/samples/${RUN}_seed123_dpm50_heldout_k64.npz

cd "${PROJECT_DIR}"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
[[ -f "${SAMPLE}" ]] || { echo "Missing sample file ${SAMPLE}" >&2; exit 1; }
mkdir -p logs/nf_conditional_om_continuous_200k results/cache/matplotlib
source "${VENV_PATH}/bin/activate"
grep -qx 'include-system-site-packages = true' "${VENV_PATH}/pyvenv.cfg" \
  || { echo "venv pyvenv.cfg changed; not touching it" >&2; exit 1; }
[[ -s "${VENV_PATH}/lib/python3.10/site-packages/00-cosmodiff-base-venv.pth" ]] \
  || { echo "00-cosmodiff-base-venv.pth missing; not touching it" >&2; exit 1; }
export PYTHONNOUSERSITE=1
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PROBE_RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/nf-om-eval-runtime.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${PROBE_RUNTIME}/sitecustomize.py"
export PYTHONPATH="${PROBE_RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/evaluate_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --run-name "${RUN}" \
  --encoder-type vgg \
  --vgg-encoder "${VGG_ENCODER}" \
  --vgg-device cpu \
  --seed 123 \
  --samples-per-cosmology 64 \
  --bootstrap 2000 \
  --output-dir "${PROJECT_DIR}/${OUTPUT_DIR}" \
  "$@"
