#!/usr/bin/env python
"""Patch a PRIVATE cosmodiff copy so classifier-free guidance with continuous labels uses a flagged null.

Ledger nf_cond_omsigc_cfg. In the stock code (cosmodiff/optim.py) continuous CFG dropout writes zeros
(train, ``torch.where(mask, torch.zeros_like(labels), labels)``) and sampling uses zeros as the null
(generate, ``null_labels = torch.zeros_like(labels)``). For z-scored labels zeros are the mean cosmology,
so the "unconditional" branch is really the mean-cosmology branch.

With ``cfg_null='flag'`` the label vector carries a last flag column: data rows are
[Omega_m_z, sigma_8_z, 0] and the null is [0, 0, 1]. This patch:

* train(): new keyword ``cfg_null`` (config key ``train.cfg_null``); raises before anything is built if
  ``conditioning == 'continuous'`` and ``cfg_dropout > 0`` and ``cfg_null`` is not 'flag'; label dropout goes
  through ``apply_cfg_label_dropout`` (same ``torch.rand(B) < cfg_dropout`` draw as the stock code; dropped rows
  become exactly the null row; every data row must have flag 0).
* generate(): new keyword ``cfg_null``; continuous guidance raises unless ``cfg_null == 'flag'``; the null is
  ``cfg_null_labels(labels)`` (= [0, ..., 0, 1]); the guided prediction goes through
  ``cfg_guided_prediction`` (same arithmetic as stock: uncond + w * (cond - uncond)).
* scripts/cosmodiff_sample.py: passes ``cfg_null = config['train'].get('cfg_null')`` to generate().

Discrete (class-label) CFG is unchanged. Refuses to run on the shared checkout; apply only to a private copy
(scripts/cosmodiff_private_runtime.py). Not idempotent on purpose: it fails if its marker is already present.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SHARED = Path("/home/jiamingp/Diffusion_model/cosmo_diffusion_main").resolve()
MARKER = "# cfg_null flag patch v1 (ledger nf_cond_omsigc_cfg)"

HELPERS = f'''

{MARKER}
_CFG_NULL_CHOICES = (None, 'flag')


def _validate_cfg_null(cfg_dropout, conditioning, cfg_null):
    """Raise unless the CFG null is well defined for this conditioning mode."""
    if cfg_null not in _CFG_NULL_CHOICES:
        raise ValueError(f"cfg_null must be one of {{_CFG_NULL_CHOICES}}, got {{cfg_null!r}}")
    if conditioning == 'continuous' and cfg_dropout > 0.0 and cfg_null != 'flag':
        raise ValueError(
            "continuous conditioning with cfg_dropout > 0 needs train.cfg_null: flag "
            "(a zero label vector is the mean cosmology for z-scored labels, not a null)"
        )
    if cfg_null == 'flag' and conditioning != 'continuous':
        raise ValueError("cfg_null='flag' is only defined for continuous conditioning")


def _check_flag_column(labels):
    if labels.shape[-1] < 2:
        raise ValueError(f"cfg_null='flag' needs a label vector with a flag column, got shape {{tuple(labels.shape)}}")
    if not bool(torch.all(labels[..., -1] == 0)):
        raise ValueError("cfg_null='flag': conditional label rows must have flag (last column) 0")


def cfg_null_labels(labels):
    """Null rows [0, ..., 0, 1] with the shape, dtype and device of ``labels``."""
    null = torch.zeros_like(labels)
    null[..., -1] = 1
    return null


def apply_cfg_label_dropout(labels, cfg_dropout, conditioning, cfg_null=None, null_token=None, generator=None):
    """Per-sample CFG label dropout (same Bernoulli draw as the stock training loop)."""
    drop_mask = torch.rand(labels.shape[0], device=labels.device, generator=generator) < cfg_dropout
    if conditioning == 'discrete':
        return torch.where(drop_mask, torch.full_like(labels, null_token), labels)
    if cfg_null != 'flag':
        raise ValueError("continuous CFG label dropout needs cfg_null='flag'")
    _check_flag_column(labels)
    mask = drop_mask.view(-1, *([1] * (labels.ndim - 1)))
    return torch.where(mask, cfg_null_labels(labels), labels)


def cfg_guided_prediction(model, images_input, timesteps, labels, null_labels, guidance_scale, cond_key):
    """Model prediction with optional CFG: uncond + w * (cond - uncond) on the raw model output."""
    pred = model(images_input, timestep=timesteps, return_dict=False, **{{cond_key: labels}})[0]
    if null_labels is not None:
        uncond_pred = model(images_input, timestep=timesteps, return_dict=False, **{{cond_key: null_labels}})[0]
        pred = uncond_pred + guidance_scale * (pred - uncond_pred)
    return pred
'''

OPTIM_EDITS = [
    # train(): new keyword at the end of the signature
    ("    sigma_log_normal: Optional[tuple] = None,\n    verbose: bool = True,\n):\n",
     "    sigma_log_normal: Optional[tuple] = None,\n    verbose: bool = True,\n    cfg_null: Optional[str] = None,\n):\n"),
    # train(): validate before anything is built
    ("    start_epoch = 0\n\n    if model is None and resume_from_checkpoint is None:\n",
     "    _validate_cfg_null(cfg_dropout, conditioning, cfg_null)\n"
     "    start_epoch = 0\n\n    if model is None and resume_from_checkpoint is None:\n"),
    # train(): dropout through the helper
    ("            if labels is not None and cfg_dropout > 0.0:\n"
     "                drop_mask = torch.rand(labels.shape[0], device=labels.device) < cfg_dropout\n"
     "                if conditioning == 'discrete':\n"
     "                    labels = torch.where(drop_mask, torch.full_like(labels, cfg_null_token), labels)\n"
     "                else:\n"
     "                    mask = drop_mask.view(-1, *([1] * (labels.ndim - 1)))\n"
     "                    labels = torch.where(mask, torch.zeros_like(labels), labels)\n",
     "            if labels is not None and cfg_dropout > 0.0:\n"
     "                labels = apply_cfg_label_dropout(labels, cfg_dropout, conditioning,\n"
     "                                                 cfg_null=cfg_null, null_token=cfg_null_token)\n"),
    # generate(): new keyword at the end of the signature
    ("    generator: Optional[torch.Generator] = None,\n) -> torch.Tensor:\n",
     "    generator: Optional[torch.Generator] = None,\n    cfg_null: Optional[str] = None,\n) -> torch.Tensor:\n"),
    # generate(): flagged null
    ("    null_labels = None\n"
     "    if labels is not None and guidance_scale is not None:\n"
     "        if conditioning == 'discrete':\n"
     "            null_token = model.config.num_class_embeds - 1\n"
     "            null_labels = torch.full_like(labels, null_token)\n"
     "        else:\n"
     "            null_labels = torch.zeros_like(labels)\n",
     "    null_labels = None\n"
     "    if labels is not None and guidance_scale is not None:\n"
     "        if conditioning == 'discrete':\n"
     "            null_token = model.config.num_class_embeds - 1\n"
     "            null_labels = torch.full_like(labels, null_token)\n"
     "        else:\n"
     "            if cfg_null != 'flag':\n"
     "                raise ValueError(\"continuous guidance needs cfg_null='flag' (zeros are the mean cosmology)\")\n"
     "            _check_flag_column(labels)\n"
     "            null_labels = cfg_null_labels(labels)\n"),
    # generate(): guided prediction through the helper
    ("            pred = model(images_input, timestep=timesteps, return_dict=False, **{cond_key: labels})[0]\n"
     "            if null_labels is not None:\n"
     "                uncond_pred = model(images_input, timestep=timesteps, return_dict=False, **{cond_key: null_labels})[0]\n"
     "                pred = uncond_pred + guidance_scale * (pred - uncond_pred)\n",
     "            pred = cfg_guided_prediction(model, images_input, timesteps, labels, null_labels,\n"
     "                                         guidance_scale, cond_key)\n"),
]

SAMPLE_EDITS = [
    ("    def cfg(key, fallback=None):\n",
     "    # cfg_null flag patch v1: the CFG null mode comes from the training section of the same config.\n"
     "    _cfg_null = None\n"
     "    if args.config is not None:\n"
     "        _cfg_null = (full_cfg.get(\"train\", {}) or {}).get(\"cfg_null\")\n\n"
     "    def cfg(key, fallback=None):\n"),
    ("            guidance_scale=cfg(\"guidance_scale\"),\n",
     "            guidance_scale=cfg(\"guidance_scale\"),\n            cfg_null=_cfg_null,\n"),
]


def apply(path: Path, edits, append: str | None = None) -> None:
    text = path.read_text()
    if MARKER in text or "cfg_null flag patch v1" in text:
        raise SystemExit(f"{path} already carries the cfg_null flag patch")
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected exactly one anchor, found {n}:\n{old}")
        text = text.replace(old, new)
    if append:
        text = text.rstrip("\n") + "\n" + append
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cosmodiff_dir", type=Path)
    a = ap.parse_args()
    root = a.cosmodiff_dir.resolve()
    if root == SHARED:
        raise SystemExit("Refusing to patch the shared cosmo_diffusion_main checkout; use a private copy.")
    optim, sample = root / "cosmodiff/optim.py", root / "scripts/cosmodiff_sample.py"
    for p in (optim, sample):
        if not p.exists():
            raise FileNotFoundError(p)
    apply(optim, OPTIM_EDITS, HELPERS)
    apply(sample, SAMPLE_EDITS)
    print("cosmo_diffusion cfg_null flag patch: patched optim.py (train, generate) and cosmodiff_sample.py")


if __name__ == "__main__":
    main()
