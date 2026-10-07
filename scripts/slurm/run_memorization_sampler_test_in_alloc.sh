#!/bin/bash
# Run the memorization sampler test inside an existing GPU allocation (e.g. via
# `srun --jobid=<jupyter job> --overlap`): the 8 sampling tasks one after another,
# then the frozen-probe evaluation. Same commands as the two sbatch files.
set -uo pipefail
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
LOG_DIR=${PROJECT_DIR}/logs/memorization_sampler_test
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
export SLURM_ARRAY_JOB_ID="inalloc_${STAMP}"
for TASK in 0 1 2 3 4 5 6 7; do
  echo "$(date +%T) sampling task ${TASK}"
  SLURM_ARRAY_TASK_ID=${TASK} bash "${PROJECT_DIR}/scripts/slurm/sample_memorization_sampler_test_array.sbatch" \
    > "${LOG_DIR}/sample_${STAMP}_${TASK}.out" 2> "${LOG_DIR}/sample_${STAMP}_${TASK}.err" \
    || { echo "task ${TASK} FAILED (see ${LOG_DIR}/sample_${STAMP}_${TASK}.err)"; exit 1; }
done
echo "$(date +%T) evaluating"
SLURM_CPUS_PER_TASK=${SLURM_CPUS_ON_NODE:-4} bash "${PROJECT_DIR}/scripts/slurm/evaluate_memorization_sampler_test.sbatch" \
  > "${LOG_DIR}/evaluate_${STAMP}.out" 2> "${LOG_DIR}/evaluate_${STAMP}.err" || { echo "evaluation FAILED"; exit 1; }
echo "$(date +%T) DONE"
