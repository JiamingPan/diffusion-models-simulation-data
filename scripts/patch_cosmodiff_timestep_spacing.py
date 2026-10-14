#!/usr/bin/env python
"""Patch a PRIVATE cosmodiff copy so cosmodiff_sample.py can override the inference timestep_spacing.

Ledger dpm50_trailing_grid_test. The stock sampler builds the inference scheduler with
``sched_cls.from_config(noise_scheduler.config)``. The project checkpoints save the training DDPMScheduler with
``timestep_spacing: leading`` (num_train_timesteps 500), so DPMSolverMultistep 50 steps runs on the leading grid
[450, 441, ..., 9]. This patch edits scripts/cosmodiff_sample.py only:

* new CLI option ``--timestep_spacing {leading,trailing,linspace}`` (default None). It is passed to
  ``sched_cls.from_config(noise_scheduler.config, timestep_spacing=...)``; None passes nothing, so the default
  behaviour is the stock one. Setting it without ``--scheduler`` is an error (it would otherwise be ignored).
* before sampling: the grid the sampler will use (set_timesteps on a deep copy of the inference scheduler with the
  same num_steps as generate()) is printed as
  ``[timestep_spacing] override=<v> spacing=<config value> n=<n> first_timestep=<t0> last_timestep=<t_end>``;
  with ``--timestep_spacing trailing`` the run exits unless the first timestep is num_train_timesteps - 1.
* after each batch: exits unless the scheduler's timesteps (set by generate()) equal that grid.
* the output npz gains ``inference_timesteps`` (int64), ``timestep_spacing`` (the scheduler config value) and
  ``timestep_spacing_override`` ("none" when not set). Sampled arrays are unchanged when the option is not set.

cosmodiff/optim.py is not touched. Refuses to run on the shared checkout; apply only to a private copy
(scripts/cosmodiff_private_runtime.py). Not idempotent on purpose: fails if its marker is already present.
Compatible with scripts/patch_cosmodiff_cfg_null_flag.py in either order (disjoint anchors).
"""
from __future__ import annotations

import argparse
from pathlib import Path

SHARED = Path("/home/jiamingp/Diffusion_model/cosmo_diffusion_main").resolve()
MARKER = "# timestep_spacing patch v1 (ledger dpm50_trailing_grid_test)"

HELPERS = f'''

{MARKER}
_TIMESTEP_SPACING_CHOICES = ("leading", "trailing", "linspace")


def timestep_spacing_kwargs(timestep_spacing):
    """Keyword arguments for sched_cls.from_config: {{}} for None (stock behaviour), else the override."""
    if timestep_spacing is None:
        return {{}}
    if timestep_spacing not in _TIMESTEP_SPACING_CHOICES:
        raise ValueError(f"timestep_spacing must be one of {{_TIMESTEP_SPACING_CHOICES}}, got {{timestep_spacing!r}}")
    return {{"timestep_spacing": timestep_spacing}}


def inference_timestep_grid(noise_scheduler, num_steps):
    """Timesteps generate() will use: set_timesteps(num_steps or num_train_timesteps) on a deep copy."""
    import copy

    probe = copy.deepcopy(noise_scheduler)
    n = num_steps if num_steps is not None else noise_scheduler.config.num_train_timesteps
    probe.set_timesteps(n)
    return [int(t) for t in probe.timesteps]


def check_timestep_grid(grid, timestep_spacing, num_train_timesteps):
    """Exit unless a trailing grid starts at the last training timestep."""
    if timestep_spacing == "trailing" and int(grid[0]) != int(num_train_timesteps) - 1:
        raise SystemExit(
            f"[timestep_spacing] trailing grid must start at {{int(num_train_timesteps) - 1}}, got {{grid[0]}}"
        )

'''

SAMPLE_EDITS = [
    # helpers before main()
    ("\n\ndef main():\n", HELPERS + "\ndef main():\n"),
    # CLI option, placed before --seed
    ('    parser.add_argument(\n        "--seed",\n',
     '    parser.add_argument(\n'
     '        "--timestep_spacing",\n'
     '        type=str,\n'
     '        choices=list(_TIMESTEP_SPACING_CHOICES),\n'
     '        default=None,\n'
     '        help="Override timestep_spacing of the inference scheduler built via --scheduler (passed to "\n'
     '             "sched_cls.from_config). Default None: the checkpoint scheduler config value (stock behaviour).",\n'
     '    )\n'
     '    parser.add_argument(\n        "--seed",\n'),
    # override passed into from_config; grid logged and checked
    ('    scheduler = cfg("scheduler")\n'
     '    if scheduler is not None:\n'
     '        import diffusers\n'
     '        sched_cls = getattr(diffusers, scheduler)\n'
     '        noise_scheduler = sched_cls.from_config(noise_scheduler.config)\n'
     '        if verbose:\n'
     '            print(f"Switched to {scheduler} for inference.")\n',
     '    scheduler = cfg("scheduler")\n'
     '    _ts_override = args.timestep_spacing  # timestep_spacing patch v1: CLI only, never read from the yaml\n'
     '    if _ts_override is not None and scheduler is None:\n'
     '        parser.error("--timestep_spacing needs --scheduler (it is passed to sched_cls.from_config)")\n'
     '    if scheduler is not None:\n'
     '        import diffusers\n'
     '        sched_cls = getattr(diffusers, scheduler)\n'
     '        noise_scheduler = sched_cls.from_config(noise_scheduler.config, **timestep_spacing_kwargs(_ts_override))\n'
     '        if verbose:\n'
     '            print(f"Switched to {scheduler} for inference.")\n'
     '    _ts_grid = inference_timestep_grid(noise_scheduler, cfg("num_steps"))\n'
     '    _ts_spacing = noise_scheduler.config.get("timestep_spacing")\n'
     '    print(f"[timestep_spacing] override={_ts_override} spacing={_ts_spacing} n={len(_ts_grid)} "\n'
     '          f"first_timestep={_ts_grid[0]} last_timestep={_ts_grid[-1]}", flush=True)\n'
     '    check_timestep_grid(_ts_grid, _ts_override, noise_scheduler.config.num_train_timesteps)\n'),
    # the grid generate() actually used
    ("        all_samples.append(samples.cpu().numpy())\n",
     "        all_samples.append(samples.cpu().numpy())\n"
     "        _ts_used = [int(t) for t in noise_scheduler.timesteps]\n"
     "        if _ts_used != _ts_grid:\n"
     "            raise SystemExit(f\"[timestep_spacing] generate() used another grid: n={len(_ts_used)} \"\n"
     "                             f\"first={_ts_used[0]} last={_ts_used[-1]}\")\n"),
    # provenance in the output file
    ('    save_kwargs = {"samples": result, "ckpt_path": np.array(ckpt_path)}\n',
     '    save_kwargs = {"samples": result, "ckpt_path": np.array(ckpt_path)}\n'
     '    save_kwargs["inference_timesteps"] = np.asarray(_ts_grid, dtype=np.int64)\n'
     '    save_kwargs["timestep_spacing"] = np.array(str(_ts_spacing))\n'
     '    save_kwargs["timestep_spacing_override"] = np.array("none" if _ts_override is None else str(_ts_override))\n'),
]


def apply(path: Path, edits) -> None:
    text = path.read_text()
    if MARKER in text or "timestep_spacing patch v1" in text:
        raise SystemExit(f"{path} already carries the timestep_spacing patch")
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected exactly one anchor, found {n}:\n{old}")
        text = text.replace(old, new)
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cosmodiff_dir", type=Path)
    a = ap.parse_args()
    root = a.cosmodiff_dir.resolve()
    if root == SHARED:
        raise SystemExit("Refusing to patch the shared cosmo_diffusion_main checkout; use a private copy.")
    sample = root / "scripts/cosmodiff_sample.py"
    if not sample.exists():
        raise FileNotFoundError(sample)
    apply(sample, SAMPLE_EDITS)
    print("cosmo_diffusion timestep_spacing patch: patched cosmodiff_sample.py (--timestep_spacing, grid log/check)")


if __name__ == "__main__":
    main()
