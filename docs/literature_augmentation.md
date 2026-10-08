# Data augmentation in diffusion-model memorization studies

Literature check, 2026-10-08 (agent read paper HTML/PDF and one code repository). **Verify every quote against the
PDF before citing it in the paper.** Context: our unconditional UNet-128 sweep found that D4 + periodic-shift
augmentation removes copying down to N = 64, while the no-augmentation transition is at N = 1024
(`results/unet128_aug_sweep/evaluate_run_v1/`, ledger `unet128_aug_sweep_eval`).

## Short answer

The papers that define the memorization-to-generalization transition in N switch augmentation **off on purpose**,
as a control. No paper checked measures how augmentation moves the transition in N.

## Table

| Paper (arXiv) | Augmentation used | Discussed as memorization factor | Key quote / location |
|---|---|---|---|
| Gu et al., 2310.02664 (TMLR 2025) | No, disabled (EDM's CIFAR-10 default is on) | Only as a confound; not among the studied factors (data size, dimension, diversity, width/depth, time embedding, skips, batch, wd, EMA, conditioning) | Sec. 2: "We train an EDM model on the CIFAR-10 dataset without applying data augmentation (to avoid any ambiguity regarding memorization)". App. B: "we disable the data augmentation to prevent any ambiguity regarding memorization." |
| Kadkhodaie et al., 2310.02557 | Not found in paper (App. A.1: noise range, batch 512, 1000 epochs). Released code `dataloader_func.py` defines `augment_training_data` (D4 rot90/flip) but `main.py` never calls it | No | Code: LabForComputationalVision/memorization_generalization_in_diffusion_models |
| Somepalli et al., 2212.03860 | Yes: horizontal flip + random crop | Only to define replication up to augmentation | Sec. 5: "We train all models with random horizontal flip and random crop augmentations." Sec. 3 discounts "minor differences in appearance that can be explained by data augmentation." |
| Somepalli et al., 2305.20086 | Not found (fine-tunes SD 2.1) | No; mitigations are caption-side | — |
| Bonnaire et al., 2505.17638 (NeurIPS 2025) | No | Only as a control | App. A.1: "To precisely control the samples seen by a model, no data augmentation is applied, and we vary the training set size n in the window [128,32768]." |
| Kamb & Ganguli, 2412.20292 | Not found (App. C: Adam, 300 epochs, no normalization layers) | Implicitly, in theory | Sec. 3.1: the equivariant score is the ideal score with "the dataset D augmented to the orbit of D under the equivariance group G… G(D) corresponds to all possible spatial translations of all images in D." Sec. 3.4: zero padding breaks equivariance; circular padding "yields more texture-like outputs". |
| Yoon et al., ICML-W 2023 (OpenReview shciCbSk9h) | Not found (full text not accessible) | Not found | — |
| Zhang et al., 2310.05264 (ICML 2024) | Main N-sweep (CIFAR-10, 2^6–2^15, UNet-64/128/256): not stated. Fine-tuning appendix: "did not involve any data augmentation." | Only as a reproducibility nuisance | Appendix: flipped generations between SD v1-3 and v1-4 "potentially a result of data augmentation introducing randomness." |
| Ho et al. DDPM, 2006.11239 | Yes: horizontal flips (all datasets except LSUN Bedroom) | No; justified by sample quality | App. B: flips "improve sample quality slightly." |
| Karras et al. EDM, 2206.00364 | Yes: non-leaky geometric pipeline (p = 12% CIFAR-10, 15% FFHQ/AFHQv2), x-flip always on; none for ImageNet-64 | As an overfitting regularizer, judged by FID | Sec. 5: "To prevent potential overfitting that often plagues diffusion models with smaller datasets…". App. F.2: "a large variety of unique training samples, preventing it from overfitting to any individual sample." Table 2, E→F: FID 1.88→1.79 (cond.), 2.05→1.97 (uncond. CIFAR-10). |
| Carlini et al., 2301.13188 | Yes: CIFAR-10 models trained on h-flipped images | Used to strengthen the attack, not as mitigation | Sec. 5.2.2 "Augmentations Improve Attacks": TPR at 0.1% FPR 7% → 44% with augmentation-averaged loss plus another trick. |
| Dar et al., 2402.01054 (3D medical LDMs) | Yes, as a test: flips + ±5° rotations, p = 0.5 each | Yes, measured | "Overall, we observed that using data augmentation reduced memorization." fastMRI: 8.7 / 6.3% memorized with augmentation (no-augmentation baselines ~25 / 31% from a less reliable extraction; check before quoting). |
| Guan et al., 2502.09434 | RandAugment, threshold-aware | Yes, measured | CIFAR-10 memorized count MQ0.5 117 → 81 when their augmentation is added to their method. |
| Hou et al. ScoreAug, 2508.07926 | Transforms the noisy input (equivariant target) | Overfitting via FID only | Abstract: "ScoreAug effectively mitigates overfitting across diverse scenarios, such as varying data scales". |

Also checked, no mention of augmentation: Marion & Wu 2605.06077, Merger & Goldt 2606.14390, Ye et al. 2511.03202, 2505.16959.

## Synthesis

* Known qualitatively: augmentation reduces memorization or overfitting (EDM states it, measures FID only;
  Dar et al. and Guan et al. measure lower memorization rates).
* The transition-in-N papers (Gu; Bonnaire) exclude augmentation explicitly "to avoid ambiguity". Our
  no-augmentation transition follows the same standard protocol.
* Theory link: Kamb & Ganguli show a translation-equivariant denoiser behaves like the ideal denoiser on the
  orbit G(D), i.e. the shift-augmented dataset; not tested experimentally there, and boundaries (zero padding) break it.
* Not found in any paper checked: a measured shift of the N-transition under exact-symmetry augmentation, compared
  with the effective-N = |G|·N expectation. That part of our result looks novel; that augmentation affects
  memorization does not.

## Framing point

For natural images only x-flips are close to label-preserving; EDM conditions on the augmentation parameters so
geometric transforms do not "leak" into samples and notes mirrored text and logos as the downside of plain flips
(App. F.2). For periodic, statistically homogeneous simulation boxes, D4 and periodic shifts are exact symmetries
of the distribution, so no conditioning is needed; this plausibly explains why image papers do not use shifts.
Caution: EDM's conditioned augmentation sends transformed copies to separate auxiliary tasks and samples only
a = 0; our unconditioned symmetric augmentation puts them in the same distribution, so EDM is not a direct precedent.
