# Notebooks

These notebooks record the analysis behind the U-Net, DiT, conditional,
augmentation, EMA, and reproducibility studies. Most expect local result tables
or sample files produced by the scripts in the repository.

## Memorization, Generalization, and Physical Statistics

- [SSCD generalizability figure](generalizability_figure.ipynb)
- [PCA generalizability figure](generalizability_figure_pca.ipynb)
- [PCA representation diagnostics](generalizability_figure_pca_diagnostics.ipynb)
- [U-Net data-size sweep and physical-statistics checks](nf_generalize_fig2_partial_quickcheck.ipynb)
- [Small-data memorization quickcheck](nf_generalize_n64_memorize_quickcheck.ipynb)
- [Reference U-Net data sweep](nf_generalize_nick_data_quickcheck.ipynb)
- [Reference sweep with SSCD and physical statistics](nf_generalize_nick_data_sscd_quickcheck.ipynb)
- [DDPM and DPM-Solver sampling comparison](nf_generalize_nick_data_dpm50_compare.ipynb)
- [Transition-scaling diagnostic](nf_generalize_scaling_diagnostic.ipynb)
- [Training-checkpoint learning process](nf_generalize_epoch_snapshots_poster.ipynb)

## Diffusion Transformers

- [DiT depth and data-size results](nf_generalize_fig2_dit_results.ipynb)
- [Audited DiT-L16 300k-500k continuation analysis](nf_generalize_fig2_dit_l16_300k_500k_analysis.ipynb)
- [Executed DiT results snapshot](nf_generalize_fig2_dit_results_executed.ipynb)

The audited continuation notebook expects the Great Lakes-only manifests,
samples, metrics, and physics arrays. Execute it on Great Lakes with:

```bash
cd /home/jiamingp/diffusion_models_repo
jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/nf_generalize_fig2_dit_l16_300k_500k_analysis.ipynb
```

## Conditional Generation and Calibration

- [Four-field class-conditional quickcheck](nf_class_conditional_fourfield_quickcheck.ipynb)
- [Continuous conditional inference](nf_conditional_inference_results.ipynb)
- [Continuous cosmology bias probe](nf_conditional_bias_probe_check.ipynb)
- [VGG cosmology recovery results](nf_conditional_bias_vgg_results.ipynb)

## Training, EMA, Augmentation, and Ablations

- [U64 run and hyperparameter inspection](inspect_run16_run12_run15.ipynb)
- [U64, U128, and U256 run inspection](inspect_run9_run10_run11.ipynb)
- [U64 full-data EMA check](nf_generalize_fig2_u64_d2p15_ema_check.ipynb)
- [Poster EMA and guidance ablations](nf_poster_ablation_appendix.ipynb)
- [Augmentation sweep quickcheck](nf_sweep_aug_quickcheck.ipynb)
- [EMA length sweep quickcheck](nf_sweep_ema_sigma_quickcheck.ipynb)
- [Initial sweep inspection](nf_sweep_inspection.ipynb)
- [Small-EMA quickcheck](nf_sweep_small_ema_quickcheck.ipynb)
- [Direct and post-hoc EMA comparison](nf_sweep_v2_ema_direct_vs_posthoc.ipynb)
- [Second sweep quickcheck](nf_sweep_v2_quickcheck.ipynb)
- [Second sweep smoke-training check](nf_sweep_v2_smoke_train_check.ipynb)

## Reproducibility and Data Validation

- [Normalization-fix and SSCD evaluation](normalization_fixes_sscd_eval.ipynb)
- [Cross-run reproducibility analysis](reproducibility_figure_analysis.ipynb)

Notebook outputs can be large. Keep generated images and tables in `results/`
rather than embedding every intermediate artifact in git.
