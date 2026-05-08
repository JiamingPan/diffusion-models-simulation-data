#!/bin/bash
set -euo pipefail

# Submit the U64/U128 reproducibility training array and, optionally, the
# matching sampling array. Run this from the repo root on Great Lakes:
#
#   bash scripts/slurm/submit_repro_u64_u128.sh
#
# Environment overrides:
#   SUBMIT_SAMPLE=0     submit training only
#   NUM_SAMPLES=512     generated samples per trained model for sampling
#   OVERWRITE=1         regenerate sample .npy files if they already exist
#   PREPARE_CONFIGS=0   skip rewriting local/fig1_lh configs

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
SUBMIT_SAMPLE=${SUBMIT_SAMPLE:-1}
PREPARE_CONFIGS=${PREPARE_CONFIGS:-1}

cd "${PROJECT_DIR}"

if [[ "${PREPARE_CONFIGS}" == "1" ]]; then
  python scripts/prepare_repro_u64_u128_configs.py --project-dir "${PROJECT_DIR}"
fi

train_job=$(sbatch --parsable scripts/slurm/train_repro_u64_u128_array.sbatch)
echo "Submitted training array: ${train_job}"

if [[ "${SUBMIT_SAMPLE}" == "1" ]]; then
  sample_job=$(sbatch --parsable --dependency=afterok:${train_job} scripts/slurm/sample_repro_u64_u128_array.sbatch)
  echo "Submitted sampling array after successful training: ${sample_job}"
else
  echo "Sampling submission disabled. Submit later with:"
  echo "  sbatch scripts/slurm/sample_repro_u64_u128_array.sbatch"
fi
