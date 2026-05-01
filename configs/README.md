# Configs

YAML files here are consumed by `nkern/cosmo_diffusion`'s
`cosmodiff_train.py`.

```text
templates/     portable starting points for CAMELS LH experiments
great_lakes/   Jiaming's Great Lakes experiment configs
```

The `great_lakes/` configs contain cluster-specific absolute paths. They are
kept as an experiment record, not as portable examples.

For a new experiment, copy a file from `templates/`, then edit:

- `data.img_path`
- `io.output_dir`
- `train.num_epochs`
- `lr_scheduler.kwargs.T_max`
- any desired model width/data-size settings
