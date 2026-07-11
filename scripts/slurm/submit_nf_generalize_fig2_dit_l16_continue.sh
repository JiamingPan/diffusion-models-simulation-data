#!/bin/bash
# Submit the four-stage DiT-L16 continuation and exact-checkpoint sampling chain.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
OVERWRITE=${OVERWRITE:-0}
SUBMIT_ANALYSIS=${SUBMIT_ANALYSIS:-1}

cd "${PROJECT_DIR}"

python scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py \
  --project-dir "${PROJECT_DIR}"

python scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --check-only

ANALYSIS_MANIFEST=${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_continue/analysis_manifest.json
if [[ "${SUBMIT_ANALYSIS}" == "1" ]]; then
  baseline_pca=$(
    MANIFEST_PATH="${ANALYSIS_MANIFEST}" SAMPLE_LABEL="dpm50" \
    OUT_PREFIX="nf_generalize_fig2_dit_l16_cont_200k_pca_full_nn" \
      sbatch -A "${ACCOUNT}" --parsable \
      scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch
  )
  baseline_sscd=$(
    MANIFEST_PATH="${ANALYSIS_MANIFEST}" SAMPLE_LABEL="dpm50" \
    OUT_PREFIX="nf_generalize_fig2_dit_l16_cont_200k_sscd_full_nn" \
      sbatch -A "${ACCOUNT}" --parsable \
      scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch
  )
  echo "baseline 200k analysis: pca=${baseline_pca%%;*} sscd=${baseline_sscd%%;*}"
fi

previous_job=""
echo "Submitting four sequential 25k-update stages; each array is limited to two GPUs."
for stage in 1 2 3 4; do
  dependency_args=()
  if [[ -n "${previous_job}" ]]; then
    dependency_args+=(--dependency="afterok:${previous_job}")
  fi

  train_job=$(
    CONTINUE_STAGE="${stage}" sbatch -A "${ACCOUNT}" --array=0-4%2 --parsable \
      "${dependency_args[@]}" \
      scripts/slurm/train_nf_generalize_fig2_dit_l16_continue_array.sbatch
  )
  train_job=${train_job%%;*}

  sample_job=$(
    CONTINUE_STAGE="${stage}" OVERWRITE="${OVERWRITE}" \
      sbatch -A "${ACCOUNT}" --array=0-4%2 --parsable \
      --dependency="afterok:${train_job}" \
      scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue_array.sbatch
  )
  sample_job=${sample_job%%;*}

  echo "stage ${stage}: train=${train_job} sample=${sample_job}"

  if [[ "${SUBMIT_ANALYSIS}" == "1" ]]; then
    total_k=$((200 + 25 * stage))
    sample_label="dpm50_cont_${total_k}k"
    pca_job=$(
      SAMPLE_LABEL="${sample_label}" \
      MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
      OUT_PREFIX="nf_generalize_fig2_dit_l16_cont_${total_k}k_pca_full_nn" \
        sbatch -A "${ACCOUNT}" --parsable \
        --dependency="afterok:${sample_job}" \
        scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch
    )
    pca_job=${pca_job%%;*}
    sscd_job=$(
      SAMPLE_LABEL="${sample_label}" \
      MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
      OUT_PREFIX="nf_generalize_fig2_dit_l16_cont_${total_k}k_sscd_full_nn" \
        sbatch -A "${ACCOUNT}" --parsable \
        --dependency="afterok:${sample_job}" \
        scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch
    )
    sscd_job=${sscd_job%%;*}
    echo "         analysis: pca=${pca_job} sscd=${sscd_job}"
  fi
  previous_job=${sample_job}
done

echo "The chain is fully sequential across stages, with at most two GPU tasks active at once."
echo "If a stage times out, re-submit only that stage; it resumes from its latest recovery checkpoint."
