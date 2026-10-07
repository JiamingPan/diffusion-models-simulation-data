#!/bin/bash
# One training task of the width-128 augmentation sweep: ARM_TAG x SLURM_ARRAY_TASK_ID (0-5 = N 64..2048).
set -euo pipefail
: "${ARM_TAG:?set ARM_TAG to d4shift, warp or d4equiv}"
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
# The tanh-norm center is a float32 mean whose last bit depends on the torch thread count;
# the verified dataset hashes were computed with 8 threads (verify job, 8 CPUs).
export OMP_NUM_THREADS=8

TASK=${SLURM_ARRAY_TASK_ID}
grep -q '"status": "VERIFIED"' "${SWEEP_LOCAL}/verification_report.json" \
  || { echo "Run verify_unet128_aug_sweep.sbatch first" >&2; exit 1; }

RUN_NAME=$(run_field "${ARM_TAG}" "${TASK}" run_name)
ARM=$(run_field "${ARM_TAG}" "${TASK}" arm)
N=$(run_field "${ARM_TAG}" "${TASK}" dataset_size)
CONFIG=$(run_field "${ARM_TAG}" "${TASK}" config)
CONFIG_SHA=$(run_field "${ARM_TAG}" "${TASK}" config_sha256)
OUTPUT_DIR=$(run_field "${ARM_TAG}" "${TASK}" output_dir)
SEED=$(run_field "${ARM_TAG}" "${TASK}" training_seed)
DATA_SHA=$("${PYTHON_BIN}" -c "import json,sys; print(json.load(open('${SWEEP_LOCAL}/data_reference.json'))['${N}']['dataset_sha256'])")
mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "Training ${RUN_NAME}: arm=${ARM} N=${N} seed=${SEED}"
echo "  config ${CONFIG} (${CONFIG_SHA:0:12}), dataset ${DATA_SHA:0:12}"
echo "  output ${OUTPUT_DIR}"
"${PYTHON_BIN}" scripts/run_unet64_three_method.py \
  --runtime-root "${COSMODIFF_DIR}" \
  --receipt "${SWEEP_LOCAL}/preflight_receipt.json" \
  --arm "${ARM}" \
  --config "${CONFIG}" \
  --expected-config-sha256 "${CONFIG_SHA}" \
  --expected-dataset-sha256 "${DATA_SHA}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --execute
