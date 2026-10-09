#!/bin/bash
# One training task of augmentation wave 2: SLURM_ARRAY_TASK_ID 0-8 indexes local/unet128_aug_wave2/runs.json.
set -euo pipefail
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_wave2_env.sh
# The tanh-norm center is a float32 mean whose last bit depends on the torch thread count;
# the verified dataset hashes were computed with 8 threads (wave-1 verify job, 8 CPUs).
export OMP_NUM_THREADS=8

TASK=${SLURM_ARRAY_TASK_ID}
RECEIPT=${RECEIPT:-${WAVE_LOCAL}/preflight_receipt.json}
grep -q '"status": "VERIFIED"' "${REF_LOCAL}/verification_report.json" \
  || { echo "wave-1 verification_report.json not VERIFIED" >&2; exit 1; }
[[ -f "${RECEIPT}" ]] || { echo "Missing preflight receipt ${RECEIPT}" >&2; exit 1; }

RUN_NAME=$(wave_field "${TASK}" run_name)
ARM=$(wave_field "${TASK}" arm)
N=$(wave_field "${TASK}" dataset_size)
CONFIG=$(wave_field "${TASK}" config)
CONFIG_SHA=$(wave_field "${TASK}" config_sha256)
OUTPUT_DIR=$(wave_field "${TASK}" output_dir)
SEED=$(wave_field "${TASK}" training_seed)
DATA_SHA=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${REF_LOCAL}/data_reference.json'))['${N}']['dataset_sha256'])")
[[ "${DATA_SHA}" == "$(wave_field "${TASK}" dataset_sha256)" ]] || { echo "dataset hash mismatch runs.json vs data_reference.json" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to reuse ${OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "Training ${RUN_NAME}: arm=${ARM} N=${N} seed=${SEED}"
echo "  config ${CONFIG} (${CONFIG_SHA:0:12}), dataset ${DATA_SHA:0:12}"
echo "  output ${OUTPUT_DIR}"
"${PYTHON_BIN}" scripts/run_unet128_aug_wave2.py \
  --runtime-root "${COSMODIFF_DIR}" \
  --receipt "${RECEIPT}" \
  --arm "${ARM}" \
  --config "${CONFIG}" \
  --expected-config-sha256 "${CONFIG_SHA}" \
  --expected-dataset-sha256 "${DATA_SHA}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --execute
