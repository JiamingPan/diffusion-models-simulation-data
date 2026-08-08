#!/bin/bash
# Submit five exact, phase-matched DiT-L16 continuation stages through 500k.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
START_STAGE=${START_STAGE:-1}
REUSE_EXISTING_MANIFEST=${REUSE_EXISTING_MANIFEST:-0}
OVERWRITE=${OVERWRITE:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2
PREPARE=${PROJECT_DIR}/scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py
MANIFEST_DIR=${PROJECT_DIR}/local/${SWEEP}
ANALYSIS_MANIFEST=${MANIFEST_DIR}/analysis_manifest.json
DATASET_TAGS=(d2p06 d2p07 d2p08 d2p09 d2p10 d2p11 d2p12 d2p13 d2p14 d2p15)

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/${SWEEP}"

if (( START_STAGE < 1 || START_STAGE > 5 )); then
  echo "START_STAGE must be between 1 and 5; got ${START_STAGE}" >&2
  exit 2
fi
if (( START_STAGE > 1 )) && [[ "${REUSE_EXISTING_MANIFEST}" != "1" ]]; then
  echo "START_STAGE>1 requires REUSE_EXISTING_MANIFEST=1" >&2
  exit 2
fi

if [[ "${REUSE_EXISTING_MANIFEST}" == "1" ]]; then
  test -s "${MANIFEST_DIR}/manifest.json"
  test -s "${ANALYSIS_MANIFEST}"
  "${PYTHON_BIN}" "${PREPARE}" --project-dir "${PROJECT_DIR}" --use-existing-manifest --check-only
else
  "${PYTHON_BIN}" "${PREPARE}" --project-dir "${PROJECT_DIR}"
fi

"${PYTHON_BIN}" "${PREPARE}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --seed-checkpoints

if (( START_STAGE > 1 )); then
  for DATASET_TAG in "${DATASET_TAGS[@]}"; do
    PREVIOUS_CHECKPOINT=$("${PYTHON_BIN}" "${PREPARE}" \
      --project-dir "${PROJECT_DIR}" \
      --use-existing-manifest \
      --dataset-tag "${DATASET_TAG}" \
      --stage "${START_STAGE}" \
      --print-field previous_expected_checkpoint)
    if [[ ! -d "${PREVIOUS_CHECKPOINT}" ]]; then
      echo "Missing restart checkpoint for ${DATASET_TAG}: ${PREVIOUS_CHECKPOINT}" >&2
      exit 1
    fi
    RUN_NAME=$("${PYTHON_BIN}" "${PREPARE}" \
      --project-dir "${PROJECT_DIR}" \
      --use-existing-manifest \
      --dataset-tag "${DATASET_TAG}" \
      --stage 1 \
      --print-field run_name)
    for COMPLETED_STAGE in $(seq 1 $((START_STAGE - 1))); do
      COMPLETED_K=$((300 + 40 * COMPLETED_STAGE))
      COMPLETED_SAMPLE="${PROJECT_DIR}/results/${SWEEP}/samples/${RUN_NAME}_seed123_dpm50_cont_${COMPLETED_K}k.npz"
      if [[ ! -s "${COMPLETED_SAMPLE}" ]]; then
        echo "Missing prior-stage sample for restart: ${COMPLETED_SAMPLE}" >&2
        echo "Restart from stage ${COMPLETED_STAGE} instead." >&2
        exit 1
      fi
    done
  done
  for COMPLETED_STAGE in $(seq 1 $((START_STAGE - 1))); do
    COMPLETED_K=$((300 + 40 * COMPLETED_STAGE))
    for FEATURE in pca sscd; do
      TABLE="${PROJECT_DIR}/results/nf_generalize_fig2_dit/tables/${SWEEP}_${COMPLETED_K}k_${FEATURE}_full_nn_metrics.csv"
      if [[ ! -s "${TABLE}" ]]; then
        echo "Missing prior-stage metric table for restart: ${TABLE}" >&2
        echo "Restart from stage ${COMPLETED_STAGE} instead." >&2
        exit 1
      fi
    done
  done
fi

PRECHECK_JOB=$(sbatch -A "${ACCOUNT}" --parsable \
  scripts/slurm/precheck_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch)
PRECHECK_JOB=${PRECHECK_JOB%%;*}
echo "precheck=${PRECHECK_JOB}"

BASELINE_SAMPLE_JOB=$(CONTINUE_STAGE=0 OVERWRITE="${OVERWRITE}" \
  sbatch -A "${ACCOUNT}" \
  --array=0-9%2 \
  --dependency="afterok:${PRECHECK_JOB}" \
  --parsable \
  scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch)
BASELINE_SAMPLE_JOB=${BASELINE_SAMPLE_JOB%%;*}
echo "audited 300k baseline samples=${BASELINE_SAMPLE_JOB}"

PREVIOUS_JOB=${PRECHECK_JOB}
ANALYSIS_JOBS=()
FINAL_TRAIN_JOB=
FINAL_SAMPLE_JOB=

BASELINE_PCA_JOB=$(MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
  SAMPLE_LABEL=dpm50_source_300k \
  OUT_PREFIX="${SWEEP}_300k_pca_full_nn" \
  sbatch -A "${ACCOUNT}" --dependency="afterok:${BASELINE_SAMPLE_JOB}" --parsable \
  scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch)
BASELINE_PCA_JOB=${BASELINE_PCA_JOB%%;*}
BASELINE_SSCD_JOB=$(MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
  SAMPLE_LABEL=dpm50_source_300k \
  OUT_PREFIX="${SWEEP}_300k_sscd_full_nn" \
  sbatch -A "${ACCOUNT}" --dependency="afterok:${BASELINE_SAMPLE_JOB}" --parsable \
  scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch)
BASELINE_SSCD_JOB=${BASELINE_SSCD_JOB%%;*}
ANALYSIS_JOBS+=("${BASELINE_PCA_JOB}" "${BASELINE_SSCD_JOB}")
echo "300k analysis: pca=${BASELINE_PCA_JOB} sscd=${BASELINE_SSCD_JOB}"

for STAGE in $(seq "${START_STAGE}" 5); do
  TRAIN_JOB=$(CONTINUE_STAGE="${STAGE}" sbatch -A "${ACCOUNT}" \
    --array=0-9%2 \
    --dependency="afterok:${PREVIOUS_JOB}" \
    --parsable \
    scripts/slurm/train_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch)
  TRAIN_JOB=${TRAIN_JOB%%;*}

  SAMPLE_JOB=$(CONTINUE_STAGE="${STAGE}" SAMPLE_MODE=DPM50 OVERWRITE="${OVERWRITE}" \
    sbatch -A "${ACCOUNT}" \
    --array=0-9%2 \
    --dependency="afterok:${TRAIN_JOB}" \
    --parsable \
    scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch)
  SAMPLE_JOB=${SAMPLE_JOB%%;*}

  TARGET_K=$((300 + 40 * STAGE))
  SAMPLE_LABEL=dpm50_cont_${TARGET_K}k
  PCA_JOB=$(MANIFEST_PATH="${ANALYSIS_MANIFEST}" SAMPLE_LABEL="${SAMPLE_LABEL}" \
    OUT_PREFIX="${SWEEP}_${TARGET_K}k_pca_full_nn" \
    sbatch -A "${ACCOUNT}" --dependency="afterok:${SAMPLE_JOB}" --parsable \
    scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch)
  PCA_JOB=${PCA_JOB%%;*}
  SSCD_JOB=$(MANIFEST_PATH="${ANALYSIS_MANIFEST}" SAMPLE_LABEL="${SAMPLE_LABEL}" \
    OUT_PREFIX="${SWEEP}_${TARGET_K}k_sscd_full_nn" \
    sbatch -A "${ACCOUNT}" --dependency="afterok:${SAMPLE_JOB}" --parsable \
    scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch)
  SSCD_JOB=${SSCD_JOB%%;*}
  ANALYSIS_JOBS+=("${PCA_JOB}" "${SSCD_JOB}")

  echo "stage ${STAGE} (${TARGET_K}k): train=${TRAIN_JOB} sample=${SAMPLE_JOB}"
  echo "  analysis: pca=${PCA_JOB} sscd=${SSCD_JOB}"
  PREVIOUS_JOB=${SAMPLE_JOB}
  FINAL_TRAIN_JOB=${TRAIN_JOB}
  FINAL_SAMPLE_JOB=${SAMPLE_JOB}
done

PHYSICS_JOB=$(sbatch -A "${ACCOUNT}" \
  --dependency="afterok:${BASELINE_SAMPLE_JOB}:${FINAL_SAMPLE_JOB}" \
  --parsable \
  scripts/slurm/analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.sbatch)
PHYSICS_JOB=${PHYSICS_JOB%%;*}
ANALYSIS_JOBS+=("${PHYSICS_JOB}")
echo "full 300k-500k physics analysis=${PHYSICS_JOB}"

DDPM_JOB=$(OVERWRITE="${OVERWRITE}" \
  sbatch -A "${ACCOUNT}" \
  --array=0-3%2 \
  --dependency="afterok:${FINAL_TRAIN_JOB}" \
  --parsable \
  scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue500k_v2_ddpm_controls.sbatch)
DDPM_JOB=${DDPM_JOB%%;*}

AUDIT_DEPENDENCY="afterok:${BASELINE_SAMPLE_JOB}:${FINAL_SAMPLE_JOB}:${DDPM_JOB}"
for JOB_ID in "${ANALYSIS_JOBS[@]}"; do
  AUDIT_DEPENDENCY+=":${JOB_ID}"
done
AUDIT_JOB=$(sbatch -A "${ACCOUNT}" \
  --dependency="${AUDIT_DEPENDENCY}" \
  --parsable \
  scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch)
AUDIT_JOB=${AUDIT_JOB%%;*}

echo "DDPM500 controls=${DDPM_JOB}; final audit=${AUDIT_JOB}"
echo "All five stages are sequential and each GPU array is limited to two tasks."
echo "Restart a failed stage with:"
echo "  START_STAGE=<1-5> REUSE_EXISTING_MANIFEST=1 bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh"
