#!/bin/bash
# Submit ten clean DiT-L16 runs directly to 300k updates.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
PYTHON_BIN=${PYTHON_BIN:-python}
OVERWRITE=${OVERWRITE:-0}
SUBMIT_ANALYSIS=${SUBMIT_ANALYSIS:-1}
REUSE_EXISTING_MANIFEST=${REUSE_EXISTING_MANIFEST:-0}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_fresh300k_v2_configs.py
MANIFEST_DIR=${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_fresh300k_v2
MANIFEST_PATH=${MANIFEST_DIR}/manifest.json
ANALYSIS_MANIFEST=${MANIFEST_DIR}/analysis_manifest.json
SAMPLE_LABEL=dpm50_fresh300k_v2
OUT_PREFIX=nf_generalize_fig2_dit_l16_fresh300k_v2

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/nf_generalize_fig2_dit_l16_fresh300k_v2"

if [[ "${REUSE_EXISTING_MANIFEST}" == "1" ]]; then
  test -f "${MANIFEST_PATH}"
  test -f "${ANALYSIS_MANIFEST}"
  echo "Reusing the frozen clean 300k manifest."
else
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" --project-dir "${PROJECT_DIR}"
fi
"${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --check-only

require_empty=0
if [[ "${REUSE_EXISTING_MANIFEST}" != "1" ]]; then
  require_empty=1
fi
precheck=$(
  REQUIRE_EMPTY="${require_empty}" sbatch -A "${ACCOUNT}" --parsable \
    scripts/slurm/precheck_nf_generalize_fig2_dit_l16_fresh300k_v2.sbatch
)
precheck=${precheck%%;*}

train=$(
  sbatch -A "${ACCOUNT}" --array=0-9%2 --parsable \
    --dependency="afterok:${precheck}" \
    scripts/slurm/train_nf_generalize_fig2_dit_l16_fresh300k_v2_array.sbatch
)
train=${train%%;*}

sample=$(
  OVERWRITE="${OVERWRITE}" sbatch -A "${ACCOUNT}" --array=0-9%2 --parsable \
    --dependency="afterok:${train}" \
    scripts/slurm/sample_nf_generalize_fig2_dit_l16_fresh300k_v2_array.sbatch
)
sample=${sample%%;*}

echo "precheck=${precheck}"
echo "train=${train}"
echo "sample=${sample}"

if [[ "${SUBMIT_ANALYSIS}" == "1" ]]; then
  pca=$(
    MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
    SAMPLE_LABEL="${SAMPLE_LABEL}" \
    OUT_PREFIX="${OUT_PREFIX}_pca_full_nn" \
      sbatch -A "${ACCOUNT}" --time=04:00:00 --parsable \
      --dependency="afterok:${sample}" \
      scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch
  )
  pca=${pca%%;*}
  sscd=$(
    MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
    SAMPLE_LABEL="${SAMPLE_LABEL}" \
    OUT_PREFIX="${OUT_PREFIX}_sscd_full_nn" \
      sbatch -A "${ACCOUNT}" --time=04:00:00 --parsable \
      --dependency="afterok:${sample}" \
      scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch
  )
  sscd=${sscd%%;*}
  audit=$(
    sbatch -A "${ACCOUNT}" --parsable \
      --dependency="afterok:${pca}:${sscd}" \
      scripts/slurm/audit_nf_generalize_fig2_dit_l16_fresh300k_v2.sbatch
  )
  audit=${audit%%;*}
  echo "analysis: pca=${pca} sscd=${sscd}"
  echo "audit=${audit}"
fi

echo "Ten clean DiT-L16 runs target 300k updates with at most two GPUs active."
echo "Each training task has 48 hours and saves complete recovery state about every 5k updates."
echo "If any task times out, rerun with:"
echo "  REUSE_EXISTING_MANIFEST=1 bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh300k_v2.sh"
