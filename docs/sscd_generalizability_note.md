# SSCD Generalizability vs P(k) Fidelity

## Paper-Style SSCD Generalizability

For a generated image \(x\), compare it to every real training image \(y_i\)
using SSCD cosine similarity:

\[
s(x) = \max_i M_{\mathrm{SSCD}}(x, y_i).
\]

With threshold \(\tau\), count \(x\) as a near-copy if

\[
s(x) > \tau.
\]

The generalizability score is

\[
GL = 1 - P(s(x) > \tau).
\]

Equivalently, for \(N_g\) generated samples:

\[
GL = \frac{1}{N_g}\sum_{j=1}^{N_g}
\mathbf{1}\left[\max_i M_{\mathrm{SSCD}}(x_j, y_i) \le \tau\right].
\]

High \(GL\) means most generated samples are not near-copies of the training
set under SSCD. Low \(GL\) means many generated samples have a close SSCD match
inside the training set.

## Why P(k) Should Not Replace SSCD Here

The power spectrum is a distribution-level physics statistic. It tells us
whether the generated fields reproduce the correct two-point clustering.

For one generated slice \(x\) and one real slice \(y\), a useful P(k) distance is:

\[
E_{P(k)}(x, y) =
\frac{1}{N_k}\sum_k
\left|\log_{10}\frac{P_x(k)}{P_y(k)}\right|.
\]

But if you compute a nearest-neighbor distance

\[
\min_i E_{P(k)}(x, y_i),
\]

that value will tend to get smaller as the training set size grows, because
there are more \(y_i\) candidates. That does not necessarily mean the model is
memorizing individual fields. It can simply mean the larger real dataset better
covers the range of valid spectra.

Use the metrics separately:

- P(k): physics fidelity of the generated distribution.
- SSCD: image-level near-copy/generalization diagnostic.
- Pixel nearest-neighbor distance: simple sanity check, but sensitive to
  translations, rotations, and small morphology changes.

## Implementation Notes

The SSCD workflow in this repo:

- Loads `sscd_disc_mixup.torchscript.pt`.
- Converts scalar CAMELS fields to grayscale RGB images.
- Resizes to `320 x 320`.
- Applies ImageNet normalization.
- Computes L2-normalized SSCD embeddings.
- Computes maximum generated-to-training cosine similarity.

For the exact paper-style score, compare against all real training slices for
that run. For a fast smoke test, you can cap the real set with `--max-real`, but
that should be labeled as approximate.
