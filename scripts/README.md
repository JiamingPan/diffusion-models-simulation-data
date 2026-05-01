# Scripts

Lightweight wrappers for training, sampling, and evaluation.

- `train_cosmodiff.py`: calls `cosmo_diffusion/scripts/cosmodiff_train.py`.
- `sample_cosmodiff.py`: loads a checkpoint and writes generated samples to `.npy`.
- `evaluate_samples.py`: computes histogram, power-spectrum, and nearest-neighbor metrics.
- `reproducibility_eval.py`: compares multiple generated sample sets.
- `slurm/great_lakes/`: Great Lakes Slurm wrappers from Jiaming's runs.
