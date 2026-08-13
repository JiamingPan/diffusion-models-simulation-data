#!/bin/bash
# Rebuild only the checkpoint-specific PCA/SSCD tables, then rerun the audit.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2
ANALYSIS_MANIFEST=${PROJECT_DIR}/local/${SWEEP}/analysis_manifest.json
SSCD_PATH=${SSCD_PATH:-/home/jiamingp/.cache/torch/hub/sscd_disc_mixup.torchscript.pt}
UPDATES=(300 340 380 420 460 500)
SAMPLE_LABELS=(dpm50_source_300k dpm50_cont_340k dpm50_cont_380k dpm50_cont_420k dpm50_cont_460k dpm50_cont_500k)

cd "${PROJECT_DIR}"
test -s "${ANALYSIS_MANIFEST}"
test -s "${SSCD_PATH}"
mkdir -p "logs/${SWEEP}" logs/nf_generalize_fig2_dit

ANALYSIS_JOBS=()
for INDEX in "${!UPDATES[@]}"; do
  UPDATE_K=${UPDATES[INDEX]}
  SAMPLE_LABEL=${SAMPLE_LABELS[INDEX]}

  PCA_JOB=$(MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
    SAMPLE_LABEL="${SAMPLE_LABEL}" \
    OUT_PREFIX="${SWEEP}_${UPDATE_K}k_pca_full_nn" \
    sbatch -A "${ACCOUNT}" --parsable \
    scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch)
  PCA_JOB=${PCA_JOB%%;*}

  SSCD_JOB=$(MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
    SAMPLE_LABEL="${SAMPLE_LABEL}" \
    OUT_PREFIX="${SWEEP}_${UPDATE_K}k_sscd_full_nn" \
    SSCD_PATH="${SSCD_PATH}" \
    sbatch -A "${ACCOUNT}" --parsable \
    scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch)
  SSCD_JOB=${SSCD_JOB%%;*}

  ANALYSIS_JOBS+=("${PCA_JOB}" "${SSCD_JOB}")
  echo "${UPDATE_K}k metrics: pca=${PCA_JOB} sscd=${SSCD_JOB}"
done

AUDIT_DEPENDENCY=afterok
for JOB_ID in "${ANALYSIS_JOBS[@]}"; do
  AUDIT_DEPENDENCY+=":${JOB_ID}"
done

AUDIT_JOB=$(sbatch -A "${ACCOUNT}" \
  --dependency="${AUDIT_DEPENDENCY}" \
  --parsable \
  scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch)
AUDIT_JOB=${AUDIT_JOB%%;*}

echo "final audit=${AUDIT_JOB}"
echo "Submitted twelve CPU metric jobs and one dependent audit."
echo "No training or sampling jobs were submitted."
