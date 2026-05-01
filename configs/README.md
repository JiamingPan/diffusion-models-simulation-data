# Configs

YAML files here are consumed by `nkern/cosmo_diffusion`'s
`cosmodiff_train.py`.

```text
templates/     portable starting points for CAMELS LH experiments
```

Cluster-specific configs with absolute paths or account/project information are
intentionally not tracked in git. Keep those locally, for example under
`local/`, which is ignored by `.gitignore`.

For a new experiment, copy a file from `templates/`, then edit:

- `data.img_path`
- `io.output_dir`
- `train.num_epochs`
- `lr_scheduler.kwargs.T_max`
- any desired model width/data-size settings
