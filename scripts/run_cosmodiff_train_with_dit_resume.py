#!/usr/bin/env python
"""Run cosmodiff training with an in-process, class-safe checkpoint loader."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import runpy
import sys
from pathlib import Path

import yaml


def import_class(qualified_name: str):
    module_name, class_name = qualified_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def load_checkpoint_preserving_class(ckpt_path: str):
    """Reconstruct exactly the diffusers class recorded in ``config.json``."""
    import diffusers

    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing diffusers config: {config_path}")
    with open(config_path) as handle:
        model_config = json.load(handle)
    class_name = model_config.get("_class_name")
    if not class_name:
        raise ValueError(f"Checkpoint {ckpt_path!r} does not record _class_name")
    try:
        model_cls = getattr(diffusers, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} requires unavailable diffusers class {class_name!r}"
        ) from exc

    model = model_cls.from_pretrained(ckpt_path)
    meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.device.type == "meta"
    ]
    if meta_parameters:
        raise RuntimeError(
            f"Checkpoint {ckpt_path!r} left meta parameters after loading: {meta_parameters[:8]}"
        )

    checkpoint_config_path = os.path.join(ckpt_path, "checkpoint_config.yaml")
    with open(checkpoint_config_path) as handle:
        checkpoint_config = yaml.safe_load(handle)

    scheduler_cls = import_class(checkpoint_config["noise_scheduler"]["class"])
    noise_scheduler = scheduler_cls.from_pretrained(ckpt_path)
    optimizer_cls = import_class(checkpoint_config["optimizer"]["class"])
    optimizer = optimizer_cls(model.parameters())
    lr_scheduler_cls = import_class(checkpoint_config["lr_scheduler"]["class"])
    lr_scheduler = lr_scheduler_cls(
        optimizer,
        **checkpoint_config["lr_scheduler"].get("kwargs", {}),
    )

    augmentations_path = os.path.join(ckpt_path, "augmentations.pkl")
    if os.path.exists(augmentations_path):
        with open(augmentations_path, "rb") as handle:
            augmentations = pickle.load(handle)
    else:
        augmentations = None

    print(
        f"Class-safe resume loader reconstructed {type(model).__name__} "
        f"from {ckpt_path}",
        flush=True,
    )
    return model, noise_scheduler, optimizer, lr_scheduler, augmentations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cosmodiff-train", required=True, type=Path)
    args, extra_args = parser.parse_known_args()

    train_script = args.cosmodiff_train.expanduser().resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Missing cosmodiff training script: {train_script}")

    from cosmodiff import utils

    # The external training entry point imports cosmodiff.optim in this same
    # process. Its resume path therefore calls this exact loader object.
    utils.load_checkpoint = load_checkpoint_preserving_class
    sys.argv = [str(train_script), "--config", args.config, *extra_args]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
