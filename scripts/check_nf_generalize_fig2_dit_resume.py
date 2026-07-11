#!/usr/bin/env python
"""Verify that cosmo_diffusion can reconstruct a saved DiT checkpoint."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from run_cosmodiff_train_with_dit_resume import load_checkpoint_preserving_class


DEFAULT_COSMODIFF_DIR = "/home/jiamingp/Diffusion_model/cosmo_diffusion_main"
EXPECTED_CLASS = "DiTTransformer2DModel"


def declared_checkpoint_class(checkpoint: Path) -> str:
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    class_name = json.loads(config_path.read_text()).get("_class_name")
    if not class_name:
        raise ValueError(f"Checkpoint {checkpoint} does not record _class_name")
    return str(class_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cosmodiff-dir", default=DEFAULT_COSMODIFF_DIR, type=Path)
    parser.add_argument("--expected-class", default=EXPECTED_CLASS)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    cosmodiff_dir = args.cosmodiff_dir.expanduser().resolve()
    declared_class = declared_checkpoint_class(checkpoint)
    if declared_class != args.expected_class:
        raise RuntimeError(
            f"Expected checkpoint class {args.expected_class}, found {declared_class} in {checkpoint}"
        )

    from cosmodiff import utils

    utils_path = Path(inspect.getsourcefile(utils)).resolve()
    if cosmodiff_dir not in utils_path.parents:
        raise RuntimeError(f"Imported cosmodiff.utils from {utils_path}, expected under {cosmodiff_dir}")

    loaded = load_checkpoint_preserving_class(str(checkpoint))
    model = loaded[0]
    optimizer = loaded[2]
    actual_class = type(model).__name__
    if actual_class != args.expected_class:
        raise RuntimeError(
            f"Resume loader reconstructed {actual_class}, expected {args.expected_class}"
        )

    meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.device.type == "meta"
    ]
    if meta_parameters:
        raise RuntimeError(f"Resume model contains meta parameters: {meta_parameters[:8]}")

    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not optimizer_parameter_ids or not optimizer_parameter_ids.issubset(model_parameter_ids):
        raise RuntimeError("Checkpoint optimizer is not bound to the reconstructed DiT parameters")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"checkpoint: {checkpoint}")
    print(f"declared class: {declared_class}")
    print(f"reconstructed class: {actual_class}")
    print(f"parameters: {parameter_count:,}")
    print("PASS: checkpoint resume loader reconstructed DiT without meta parameters.")


if __name__ == "__main__":
    main()
