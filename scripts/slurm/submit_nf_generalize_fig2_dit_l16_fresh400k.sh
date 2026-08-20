#!/bin/bash
# Submit ten fresh DiT-L16 runs through 400k, with exact milestone evaluation.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
PYTHON_BIN=${PYTHON_BIN:-python}
OVERWRITE=${OVERWRITE:-0}
SUBMIT_ANALYSIS=${SUBMIT_ANALYSIS:-1}
START_STAGE=${START_STAGE:-1}
REUSE_EXISTING_MANIFEST=${REUSE_EXISTING_MANIFEST:-0}
PREPARE_SCRIPT=scripts/prepare_nf_generalize_fig2_dit_l16_fresh400k_configs.py
MANIFEST_DIR=${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_fresh400k
ANALYSIS_MANIFEST=${MANIFEST_DIR}/analysis_manifest.json

cd "${PROJECT_DIR}"

if (( START_STAGE < 1 || START_STAGE > 16 )); then
  echo "START_STAGE must be between 1 and 16; got ${START_STAGE}" >&2
  exit 1
fi
if [[ "${REUSE_EXISTING_MANIFEST}" == "1" ]]; then
  test -f "${MANIFEST_DIR}/manifest.json"
  test -f "${ANALYSIS_MANIFEST}"
  echo "Reusing the frozen fresh-sweep manifest."
else
  if [[ "${START_STAGE}" != "1" ]]; then
    echo "START_STAGE>1 requires REUSE_EXISTING_MANIFEST=1." >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" --project-dir "${PROJECT_DIR}"
fi

"${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --check-only

require_empty=0
if [[ "${START_STAGE}" == "1" && "${REUSE_EXISTING_MANIFEST}" != "1" ]]; then
  require_empty=1
fi
precheck=$(
  REQUIRE_EMPTY="${require_empty}" sbatch -A "${ACCOUNT}" --parsable \
    scripts/slurm/precheck_nf_generalize_fig2_dit_l16_fresh400k.sbatch
)
precheck=${precheck%%;*}
echo "fresh 400k precheck: ${precheck}"

previous_job=${precheck}
echo "Submitting 16 sequential 25k stages for all ten data sizes."
echo "Each training and sampling array is limited to two GPUs."
for stage in $(seq 1 16); do
  if (( stage < START_STAGE )); then
    continue
  fi

  train_job=$(
    TRAIN_STAGE="${stage}" sbatch -A "${ACCOUNT}" --array=0-9%2 --parsable \
      --dependency="afterok:${previous_job}" \
      scripts/slurm/train_nf_generalize_fig2_dit_l16_fresh400k_array.sbatch
  )
  train_job=${train_job%%;*}
  previous_job=${train_job}
  total_k=$((25 * stage))
  echo "stage ${stage}/16 (${total_k}k): train=${train_job}"

  if [[ "${total_k}" != "200" \
    && "${total_k}" != "300" \
    && "${total_k}" != "400" ]]; then
    continue
  fi

  sample_label="dpm50_fresh_${total_k}k"
  sample_job=$(
    SAMPLE_STAGE="${stage}" OVERWRITE="${OVERWRITE}" \
      sbatch -A "${ACCOUNT}" --array=0-9%2 --parsable \
      --dependency="afterok:${train_job}" \
      scripts/slurm/sample_nf_generalize_fig2_dit_l16_fresh400k_array.sbatch
  )
  sample_job=${sample_job%%;*}
  previous_job=${sample_job}
  echo "                   sample=${sample_job} label=${sample_label}"

  if [[ "${SUBMIT_ANALYSIS}" == "1" ]]; then
    prefix="nf_generalize_fig2_dit_l16_fresh400k_${total_k}k"
    pca_job=$(
      MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
      SAMPLE_LABEL="${sample_label}" \
      OUT_PREFIX="${prefix}_pca_full_nn" \
        sbatch -A "${ACCOUNT}" --time=04:00:00 --parsable \
        --dependency="afterok:${sample_job}" \
        scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch
    )
    pca_job=${pca_job%%;*}
    sscd_job=$(
      MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
      SAMPLE_LABEL="${sample_label}" \
      OUT_PREFIX="${prefix}_sscd_full_nn" \
        sbatch -A "${ACCOUNT}" --time=04:00:00 --parsable \
        --dependency="afterok:${sample_job}" \
        scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch
    )
    sscd_job=${sscd_job%%;*}
    previous_job="${pca_job}:${sscd_job}"
    echo "                   analysis: pca=${pca_job} sscd=${sscd_job}"

    audit_job=$(
      AUDIT_UPDATES="$((total_k * 1000))" \
        sbatch -A "${ACCOUNT}" --parsable \
        --dependency="afterok:${pca_job}:${sscd_job}" \
        scripts/slurm/audit_nf_generalize_fig2_dit_l16_fresh400k.sbatch
    )
    audit_job=${audit_job%%;*}
    previous_job=${audit_job}
    echo "                   audit=${audit_job}"
  fi
done

echo "Fresh DiT-L16 plan targets 400k updates for every data size 2^6 through 2^15."
echo "The 200k result is the equal-budget comparison, 300k is intermediate, and 400k is final."
echo "To restart from a failed stage:"
echo "  START_STAGE=<stage> REUSE_EXISTING_MANIFEST=1 bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_fresh400k.sh"
