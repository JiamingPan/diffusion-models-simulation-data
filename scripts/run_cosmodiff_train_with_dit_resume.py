#!/usr/bin/env python
"""Run cosmodiff training to one exact checkpoint under either resume API."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import pickle
import re
import runpy
import sys
import textwrap
from pathlib import Path

import yaml


CHECKPOINT_RE = re.compile(r"checkpoint-epoch-(\d+)$")


def checkpoint_epoch(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def latest_checkpoint(checkpoint_dir: Path) -> tuple[Path, int]:
    candidates = [
        (path, epoch)
        for path in checkpoint_dir.glob("checkpoint-epoch-*")
        if path.is_dir() and (epoch := checkpoint_epoch(path)) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-epoch-* directories under {checkpoint_dir}")
    return max(candidates, key=lambda item: item[1])


def validate_resume_target(
    checkpoint_dir: Path, target_checkpoint: Path
) -> tuple[Path, int, int]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    target_checkpoint = target_checkpoint.expanduser()
    target_epoch = checkpoint_epoch(target_checkpoint)
    if target_epoch is None:
        raise ValueError(f"Invalid exact target checkpoint name: {target_checkpoint}")
    if target_checkpoint.parent.resolve() != checkpoint_dir:
        raise ValueError(
            f"Exact target {target_checkpoint} is not under checkpoint directory {checkpoint_dir}"
        )

    current, current_epoch = latest_checkpoint(checkpoint_dir)
    if current_epoch > target_epoch:
        raise ValueError(
            f"Latest clean checkpoint epoch {current_epoch} is beyond exact target "
            f"epoch {target_epoch}; refusing contaminated continuation directory"
        )
    return current, current_epoch, target_epoch


def epoch_argument(start_epoch: int, target_epoch: int, semantics: str) -> int:
    """Convert an inclusive target epoch to the installed trainer's argument."""
    if start_epoch > target_epoch + 1:
        raise ValueError(
            f"Resume start epoch {start_epoch} is beyond exact target epoch {target_epoch}"
        )
    if semantics == "absolute":
        return target_epoch + 1
    if semantics == "additional":
        return target_epoch + 1 - start_epoch
    raise ValueError(f"Unknown cosmodiff epoch semantics: {semantics!r}")


def _expanded_expression(node: ast.AST, assignments: dict[str, ast.AST]) -> ast.AST:
    seen: set[str] = set()
    while isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        seen.add(node.id)
        node = assignments[node.id]
    return node


def detect_epoch_semantics(train_fn) -> str:
    """Detect whether ``num_epochs`` is an absolute end or an added duration."""
    source = textwrap.dedent(inspect.getsource(train_fn))
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "start_epoch"
        ):
            continue
        end = _expanded_expression(node.args[1], assignments)
        names = {child.id for child in ast.walk(end) if isinstance(child, ast.Name)}
        if "start_epoch" in names and "num_epochs" in names:
            return "additional"
        if "num_epochs" in names and "start_epoch" not in names:
            return "absolute"

    raise RuntimeError(
        "Could not determine cosmodiff train epoch semantics from its range(start_epoch, ...) loop"
    )


def install_exact_target_adapter(optim, *, expected_start_epoch: int, target_epoch: int) -> str:
    original_train = optim.train
    semantics = detect_epoch_semantics(original_train)

    def train_to_exact_target(*args, **kwargs):
        signature = inspect.signature(original_train)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if "start_epoch" not in bound.arguments or "num_epochs" not in bound.arguments:
            raise RuntimeError("cosmodiff.optim.train must expose start_epoch and num_epochs")
        start_epoch = int(bound.arguments["start_epoch"])
        if start_epoch != expected_start_epoch:
            raise RuntimeError(
                f"External trainer selected start epoch {start_epoch}, expected "
                f"{expected_start_epoch} from the clean checkpoint directory"
            )
        runtime_argument = epoch_argument(start_epoch, target_epoch, semantics)
        bound.arguments["num_epochs"] = runtime_argument
        print(
            "Exact-target epoch adapter: "
            f"semantics={semantics} start_epoch={start_epoch} "
            f"target_epoch={target_epoch} num_epochs_argument={runtime_argument}",
            flush=True,
        )
        return original_train(*bound.args, **bound.kwargs)

    optim.train = train_to_exact_target
    return semantics


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
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    args, extra_args = parser.parse_known_args()

    train_script = args.cosmodiff_train.expanduser().resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Missing cosmodiff training script: {train_script}")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    target_checkpoint = args.target_checkpoint.expanduser()
    current, current_epoch, target_epoch = validate_resume_target(
        checkpoint_dir, target_checkpoint
    )
    if current_epoch == target_epoch:
        print(f"Exact target already exists; no training needed: {current}", flush=True)
        return

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    configured_output_dir = Path(config["io"]["output_dir"]).expanduser().resolve()
    if configured_output_dir != checkpoint_dir:
        raise ValueError(
            f"Config output directory {configured_output_dir} does not match isolated "
            f"checkpoint directory {checkpoint_dir}"
        )

    from cosmodiff import optim, utils

    # The external training entry point imports cosmodiff.optim in this same
    # process. Its resume path therefore calls this exact loader object.
    utils.load_checkpoint = load_checkpoint_preserving_class
    install_exact_target_adapter(
        optim,
        expected_start_epoch=current_epoch + 1,
        target_epoch=target_epoch,
    )
    sys.argv = [str(train_script), "--config", args.config, *extra_args]
    runpy.run_path(str(train_script), run_name="__main__")
    if not target_checkpoint.is_dir():
        raise RuntimeError(
            f"External trainer exited without exact target checkpoint: {target_checkpoint}"
        )


if __name__ == "__main__":
    main()
