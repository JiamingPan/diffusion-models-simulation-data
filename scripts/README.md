# Scripts

Lightweight wrappers for training, sampling, and evaluation.

- `train_cosmodiff.py`: calls `cosmo_diffusion/scripts/cosmodiff_train.py`.
- `sample_cosmodiff.py`: loads a checkpoint and writes generated samples to `.npy`.
- `evaluate_samples.py`: computes histogram, power-spectrum, and nearest-neighbor metrics.
- `reproducibility_eval.py`: compares multiple generated sample sets.
- `plot_reproducibility_figure.py`: makes a Figure-1-style pairwise reproducibility plot from a manifest of generated sample files.
- `slurm/train_template.sbatch`: sanitized Slurm template. Copy it into an ignored local path and edit account, partition, paths, and walltime there.
