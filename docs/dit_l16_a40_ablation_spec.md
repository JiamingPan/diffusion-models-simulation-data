# L16 A40 initialization / patch-size ablation

Test three fresh seed-123 runs at N₂D=256 and exactly 300k nominal optimizer
updates: patch8/zero-modulation-output, patch4/native, patch4/zero-modulation-output.
Use the existing fresh-300k patch8/native run as the baseline. Preserve its actual
YAML recipe; only initialization, patch size, output identity and checkpoint
cadence may differ. No continuation, LR/loss/precision changes or depth sweep.

Match the native select/zthin/log/normalization order; never change training
tensors to match the legacy evaluation reader. Require an actual-data CPU
validation receipt for all three arms before GPU training, and evaluate new
samples and the saved baseline against the same retained-slice reference.

Run the bounded CPU patch-memorization Test A on saved matrix DPM50 samples.
Do not run the supplied Test B. Evaluate new raw-weight checkpoints using DPM++
order2/50 and the matrix's saved 128 initial noise fields. Do not infer physical
invalidity from low training cosine; inspect separate population spectra,
patch-boundary diagnostics on both 4/8-pixel grids, and maps.

Completion requires all three full checkpoint validations, finite first optimizer
steps, paired sample artifacts and a qualified comparison with the baseline.
Do not stop at submission, a Slurm exit code, or a missing/nonfinite quality score.
Continue through verification within the approved action's scope; recovery or
additional runs require another preview/approval. Local CPU tests do not prove
full-width fp16 A40 memory fit or wall time.

Detailed [runtime and verification rules](dit_l16_a40_ablation_runtime.md).
External actions remain human-controlled: show exact targets, commands/payload,
cost and side effects, then stop for fresh APPROVE PUSH / APPROVE RUN. Never reuse
previous approvals. Preserve old checkpoints, C4, main and unrelated work.
