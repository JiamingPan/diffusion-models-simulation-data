#!/usr/bin/env python
"""Run cosmodiff training to one exact checkpoint under either resume API."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
import pickle
import random
import re
import runpy
import sys
import textwrap
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.torch_compat import install_torch_backend_compat


install_torch_backend_compat(entry_point=__name__)


CHECKPOINT_RE = re.compile(r"checkpoint-epoch-(\d+)$")


def normalize_posthoc_ema_checkpoint_state(
    state: dict[str, object], *, expected_step: int
) -> dict[str, object]:
    """Restore an exact integer step after ema-pytorch's checkpoint dtype cast."""
    import torch

    if not isinstance(state, dict):
        raise ValueError("EMA snapshot is not a state mapping")
    stored_step = state.get("step")
    if not isinstance(stored_step, torch.Tensor) or stored_step.numel() != 1:
        raise ValueError("EMA snapshot step must be one tensor scalar")
    expected_as_stored = torch.tensor(
        float(expected_step),
        dtype=stored_step.dtype,
        device="cpu",
    ).reshape(stored_step.shape)
    actual_stored = stored_step.detach().cpu()
    if not torch.equal(actual_stored, expected_as_stored):
        raise ValueError(
            "EMA snapshot step does not match the filename after checkpoint "
            f"dtype conversion: stored={actual_stored.item()!r}, "
            f"expected={expected_as_stored.item()!r}"
        )
    normalized = dict(state)
    normalized["step"] = torch.full(
        stored_step.shape,
        int(expected_step),
        dtype=torch.long,
        device="cpu",
    )
    return normalized


def restore_random_states(checkpoint_dir: Path) -> tuple[str, ...]:
    """Restore the saved Python, NumPy, torch CPU, and torch CUDA RNG states."""
    import numpy as np
    import torch

    path = checkpoint_dir / "random_states_0.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"A scientific continuation requires saved RNG state: {path}"
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Saved RNG state is not a mapping: {path}")

    required = ("random_state", "numpy_random_seed", "torch_manual_seed")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Saved RNG state {path} is missing keys: {', '.join(missing)}")

    random.setstate(state["random_state"])
    np.random.set_state(state["numpy_random_seed"])
    torch.set_rng_state(state["torch_manual_seed"].cpu())
    restored = ["python", "numpy", "torch_cpu"]

    cuda_state = state.get("torch_cuda_manual_seed")
    if cuda_state and torch.cuda.is_available():
        if isinstance(cuda_state, torch.Tensor):
            cuda_state = [cuda_state]
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])
        restored.append("torch_cuda")

    print(
        f"Restored RNG state from {path}: {', '.join(restored)}",
        flush=True,
    )
    return tuple(restored)


def checkpoint_epoch(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


REQUIRED_RESUME_FILES = (
    "config.json",
    "checkpoint_config.yaml",
    "random_states_0.pkl",
)
REQUIRED_RESUME_ALTERNATIVES = {
    "noise scheduler": ("scheduler_config.json", "noise_scheduler.pkl"),
    "gradient scaler": ("scaler.pt", "scaler.bin"),
}
MODEL_WEIGHT_FILES = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)


def restore_posthoc_ema_state(
    ema,
    checkpoint_dir: Path,
    *,
    expected_step: int,
    expected_sigma_rels: list[float] | tuple[float, ...],
    expected_burn_in: int,
) -> dict[str, object]:
    """Restore every post-hoc EMA profile from one exact checkpoint step."""
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    ema_metadata = validate_checkpoint_ema_metadata(
        checkpoint_dir,
        expected_sigma_rels=expected_sigma_rels,
        expected_burn_in=expected_burn_in,
    )
    ema_dir = checkpoint_dir / "ema"
    profiles = list(getattr(ema, "ema_models", ()))
    if not profiles:
        raise ValueError("PostHocEMA exposes no EMA profiles to restore")
    if not ema_dir.is_dir():
        raise FileNotFoundError(f"Missing post-hoc EMA directory: {ema_dir}")

    loaded_paths: list[str] = []
    for profile_index, profile in enumerate(profiles):
        path = ema_dir / f"{profile_index}.{int(expected_step)}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                "Missing exact post-hoc EMA profile snapshot: " + str(path)
            )
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        state = normalize_posthoc_ema_checkpoint_state(
            state,
            expected_step=expected_step,
        )
        profile.load_state_dict(state, strict=True)
        actual_step = int(profile.step.item())
        if actual_step != int(expected_step):
            raise ValueError(
                f"EMA profile {profile_index} restored step {actual_step}, "
                f"expected {expected_step}"
            )
        if not bool(profile.initted.item()):
            raise ValueError(f"EMA profile {profile_index} is not initialized")
        loaded_paths.append(str(path))

    return {
        "step": int(expected_step),
        "profiles": len(profiles),
        "sigma_rels": ema_metadata["ema_sigma_rels"],
        "burn_in": ema_metadata["ema_burn_in"],
        "snapshots": loaded_paths,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_completed_noop_audit(
    path: Path,
    *,
    checkpoint: Path,
    run_name: str | None,
    code_revision: str | None,
) -> None:
    """Record that a retry found its exact target already complete."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite resume audit: {path}")
    _write_json_atomic(
        path,
        {
            "status": "already_complete_no_training",
            "checkpoint": str(Path(checkpoint).resolve()),
            "run_name": run_name,
            "code_revision": code_revision,
            "first_resumed_loss": None,
            "ema_restore": None,
        },
    )


def seed_training_rng(seed: int) -> None:
    """Set only the post-checkpoint Python, NumPy, and torch RNG streams."""
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resume_seed_for_checkpoint(
    requested_seed: int | None,
    current_checkpoint: Path,
    seed_origin_checkpoint: Path | None,
) -> int | None:
    """Apply a new seed once, never again when recovering a partial stage."""
    if requested_seed is None:
        return None
    if seed_origin_checkpoint is None:
        raise ValueError("A requested resume seed requires its exact origin checkpoint")
    if Path(current_checkpoint).resolve() != Path(seed_origin_checkpoint).resolve():
        return None
    return int(requested_seed)


def install_constant_label_adapter(utils_module) -> None:
    """Honor data.constant_label in memory without mutating external source."""
    original_parse_config_data = utils_module.parse_config_data

    def parse_config_data_with_constant_label(config):
        import torch

        def describe(labels):
            tensor = torch.as_tensor(labels)
            length = len(labels) if tensor.ndim > 0 else 0
            unique = [value.item() for value in torch.unique(tensor.detach().cpu())]
            return tensor, length, unique

        def log_path(path, labels):
            tensor, length, unique = describe(labels)
            print(
                f"[constant-label] path={path} dtype={tensor.dtype} "
                f"length={length} unique={unique}",
                flush=True,
            )

        output = original_parse_config_data(config)
        constant_label = config.get("data", {}).get("constant_label")
        dataset = output.get("data")
        if constant_label is not None:
            if dataset is None or not hasattr(dataset, "labels") or not hasattr(dataset, "arrays"):
                raise RuntimeError(
                    "data.constant_label requires an ArrayDataset-like object with arrays and labels"
                )
            expected_length = len(dataset.arrays)
            if dataset.labels is None:
                dataset.labels = torch.full(
                    (expected_length,),
                    int(constant_label),
                    dtype=torch.long,
                    device=dataset.arrays.device,
                )
                log_path("legacy_injected", dataset.labels)
                return output

            labels, label_length, _unique = describe(dataset.labels)
            matches_requested_constant = (
                label_length == expected_length
                and labels.numel() == expected_length
                and bool(torch.all(labels == int(constant_label)).item())
            )
            if matches_requested_constant:
                log_path("existing_constant_noop", dataset.labels)
            else:
                log_path("conflict_refused", dataset.labels)
                raise RuntimeError(
                    "Refusing to replace existing dataset labels with "
                    "data.constant_label: labels differ in length or value"
                )
        return output

    utils_module.parse_config_data = parse_config_data_with_constant_label


def install_exact_checkpoint_finder(
    utils_module,
    *,
    checkpoint_dir: Path,
    checkpoint: Path,
) -> None:
    """Make the external entry point ignore newer half-written directories."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint = Path(checkpoint).resolve()

    def find_exact_checkpoint(output_dir):
        actual_dir = Path(output_dir).expanduser().resolve()
        if actual_dir != checkpoint_dir:
            raise ValueError(
                f"External trainer searched {actual_dir}, expected {checkpoint_dir}"
            )
        return str(checkpoint)

    utils_module.find_latest_checkpoint = find_exact_checkpoint


def install_seed_restart_accelerator_hooks(
    accelerator_cls,
    *,
    checkpoint: Path,
    resume_seed: int | None,
    source_updates: int,
    source_microbatches: int,
    audit_path: Path,
    audit_context: dict[str, object],
) -> None:
    """Optionally reseed after restore, then audit the first real loss."""
    checkpoint = Path(checkpoint).resolve()
    audit_path = Path(audit_path)
    original_load_state = accelerator_cls.load_state
    original_backward = accelerator_cls.backward
    original_log = accelerator_cls.log
    original_save_state = getattr(accelerator_cls, "save_state", None)
    audit = {
        **audit_context,
        "checkpoint": str(checkpoint),
        "resume_seed": None if resume_seed is None else int(resume_seed),
        "rng_mode": "checkpoint_state" if resume_seed is None else "new_seed",
        "source_updates": int(source_updates),
        "source_microbatches": int(source_microbatches),
        "first_resumed_optimizer_step": int(source_updates) + 1,
        "first_resumed_microbatch_step": int(source_microbatches) + 1,
    }
    state = {"loaded": False, "first_loss_written": False}

    def load_state_then_reseed(self, input_dir, *args, **kwargs):
        actual = Path(input_dir).resolve()
        if actual != checkpoint:
            raise ValueError(
                f"Accelerate loaded {actual}, expected seed-restart checkpoint {checkpoint}"
            )
        result = original_load_state(self, input_dir, *args, **kwargs)
        if resume_seed is not None:
            seed_training_rng(resume_seed)
        state["loaded"] = True
        _write_json_atomic(audit_path, audit)
        return result

    def backward_and_audit(self, loss, *args, **kwargs):
        if not state["loaded"]:
            raise RuntimeError("Backward occurred before checkpoint state was restored")
        if not state["first_loss_written"]:
            audit["first_resumed_loss"] = float(loss.detach().item())
            _write_json_atomic(audit_path, audit)
            state["first_loss_written"] = True
        return original_backward(self, loss, *args, **kwargs)

    def log_with_absolute_step(self, values, *args, **kwargs):
        if state["loaded"] and kwargs.get("step") is not None:
            kwargs["step"] = int(source_microbatches) + int(kwargs["step"])
        return original_log(self, values, *args, **kwargs)

    accelerator_cls.load_state = load_state_then_reseed
    accelerator_cls.backward = backward_and_audit
    accelerator_cls.log = log_with_absolute_step
    if original_save_state is not None:
        original_burn_in = int(audit_context["original_ema_burn_in"])

        def save_state_with_absolute_ema_metadata(self, output_dir, *args, **kwargs):
            result = original_save_state(self, output_dir, *args, **kwargs)
            config_path = Path(output_dir) / "checkpoint_config.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Trainer save_state did not leave checkpoint metadata: {config_path}"
                )
            checkpoint_config = yaml.safe_load(config_path.read_text())
            checkpoint_config["ema_burn_in"] = original_burn_in
            checkpoint_config["resume_effective_ema_burn_in"] = 0
            checkpoint_config["ema_state_restored"] = True
            temporary = config_path.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(checkpoint_config, sort_keys=False))
            temporary.replace(config_path)
            return result

        accelerator_cls.save_state = save_state_with_absolute_ema_metadata


def install_seed_restart_ema_factory(
    ema_module,
    *,
    checkpoint: Path,
    expected_step: int,
    expected_sigma_rels: list[float] | tuple[float, ...],
    expected_burn_in: int,
    audit_path: Path | None = None,
) -> None:
    """Replace the process-local PostHocEMA factory with an exact-state loader."""
    original_posthoc_ema = ema_module.PostHocEMA

    def restored_posthoc_ema(*args, **kwargs):
        ema = original_posthoc_ema(*args, **kwargs)
        report = restore_posthoc_ema_state(
            ema,
            checkpoint,
            expected_step=expected_step,
            expected_sigma_rels=expected_sigma_rels,
            expected_burn_in=expected_burn_in,
        )
        ema._seed_restart_report = report
        if audit_path is not None:
            audit_file = Path(audit_path)
            audit = json.loads(audit_file.read_text()) if audit_file.exists() else {}
            audit["ema_restore"] = report
            _write_json_atomic(audit_file, audit)
        return ema

    ema_module.PostHocEMA = restored_posthoc_ema


def ema_step_for_current_checkpoint(
    *,
    stage_start_ema_step: int,
    stage_start_epoch: int,
    current_epoch: int,
    optimizer_steps_per_epoch: int,
    microbatches_per_optimizer_step: int,
) -> int:
    """Advance a stage-start EMA step when recovering a partial stage."""
    elapsed_epochs = int(current_epoch) - int(stage_start_epoch)
    if elapsed_epochs < 0:
        raise ValueError("current checkpoint precedes the required stage start")
    return int(stage_start_ema_step) + (
        elapsed_epochs
        * int(optimizer_steps_per_epoch)
        * int(microbatches_per_optimizer_step)
    )


def build_seed_restart_context(
    config: dict,
    *,
    checkpoint_epoch: int,
    optimizer_steps_per_epoch: int,
    resume_ema_step: int,
    resume_seed: int | None,
    run_name: str,
) -> dict[str, object]:
    """Build and validate the immutable state contract for a seed restart."""
    steps_per_epoch = int(optimizer_steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError("optimizer_steps_per_epoch must be positive")
    source_updates = (int(checkpoint_epoch) + 1) * steps_per_epoch

    train = config.get("train", {})
    ema_sigma_rels = train.get("ema_sigma_rels")
    if not ema_sigma_rels:
        raise ValueError("Seed restart requires enabled post-hoc EMA profiles")
    ema_update_every = int(train.get("ema_update_every", 1))
    if ema_update_every != 1:
        raise ValueError(
            "Exact seed restart currently requires ema_update_every=1; "
            f"found {ema_update_every}"
        )
    ema_burn_in = int(train.get("ema_burn_in", 0))
    accumulation = int(train.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    source_microbatches = source_updates * accumulation
    calculated_ema_step = source_microbatches - ema_burn_in
    if source_microbatches < ema_burn_in:
        raise ValueError(
            f"Checkpoint has {source_microbatches} microbatches, "
            f"before EMA burn-in {ema_burn_in}"
        )
    if int(resume_ema_step) != calculated_ema_step:
        raise ValueError(
            "Resume EMA step disagrees with optimizer/microbatch clocks: "
            f"explicit={resume_ema_step}, calculated={calculated_ema_step}, "
            f"optimizer_updates={source_updates}, accumulation={accumulation}"
        )

    data = config.get("data", {})
    data_seed = data.get("seed")
    if data_seed is not None:
        raise ValueError(
            "This seed-restart workflow requires the frozen first-n subset "
            f"contract data.seed=None; found {data_seed!r}"
        )
    paths = data.get("img_path", [])
    counts = data.get("n_samples", [])
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    if not isinstance(counts, (list, tuple)):
        counts = [counts] * len(paths)
    if len(paths) != len(counts) or not paths:
        raise ValueError("data.img_path and data.n_samples must be aligned nonempty lists")
    sources = []
    for path, count in zip(paths, counts):
        if count is None or int(count) <= 0:
            raise ValueError(f"Invalid frozen n_samples={count!r} for {path}")
        count = int(count)
        sources.append(
            {
                "img_path": str(path),
                "n_samples": count,
                "volume_indices": list(range(count)),
            }
        )
    subset = {
        "seed": None,
        "selection": "first_n",
        "reshape": data.get("reshape"),
        "zthin": data.get("zthin"),
        "label_path": data.get("label_path"),
        "constant_label": data.get("constant_label"),
        "normalization": data.get("normalization"),
        "norm_kwargs": data.get("norm_kwargs"),
        "transform": data.get("transform"),
        "sources": sources,
    }
    subset_bytes = json.dumps(
        subset, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    return {
        "run_name": str(run_name),
        "resume_seed": None if resume_seed is None else int(resume_seed),
        "rng_mode": "checkpoint_state" if resume_seed is None else "new_seed",
        "checkpoint_epoch": int(checkpoint_epoch),
        "optimizer_steps_per_epoch": steps_per_epoch,
        "source_updates": source_updates,
        "microbatches_per_optimizer_step": accumulation,
        "source_microbatches": source_microbatches,
        "first_resumed_optimizer_step": source_updates + 1,
        "first_resumed_microbatch_step": source_microbatches + 1,
        "original_ema_burn_in": ema_burn_in,
        "ema_update_every": ema_update_every,
        "ema_sigma_rels": [float(value) for value in ema_sigma_rels],
        "expected_ema_step": int(resume_ema_step),
        "training_subset": subset,
        "training_subset_sha256": hashlib.sha256(subset_bytes).hexdigest(),
    }


def checkpoint_optimizer_summary(checkpoint_dir: Path) -> dict[str, object]:
    """Read compact optimizer/LR evidence without mutating checkpoint objects."""
    checkpoint_dir = Path(checkpoint_dir)
    layout = checkpoint_training_state_layout(checkpoint_dir)
    if layout == "legacy":
        with (checkpoint_dir / "optimizer.pkl").open("rb") as handle:
            optimizer = pickle.load(handle)
        with (checkpoint_dir / "lr_scheduler.pkl").open("rb") as handle:
            scheduler = pickle.load(handle)
        state = optimizer.state_dict()
        optimizer_class = type(optimizer).__name__
        scheduler_class = type(scheduler).__name__
        scheduler_last_epoch = int(getattr(scheduler, "last_epoch", -1))
        scheduler_last_lr = [float(value) for value in scheduler.get_last_lr()]
    elif layout == "native":
        import torch

        try:
            state = torch.load(
                checkpoint_dir / "optimizer.bin",
                map_location="cpu",
                weights_only=True,
            )
            scheduler_state = torch.load(
                checkpoint_dir / "scheduler.bin",
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state = torch.load(checkpoint_dir / "optimizer.bin", map_location="cpu")
            scheduler_state = torch.load(
                checkpoint_dir / "scheduler.bin", map_location="cpu"
            )
        checkpoint_config = yaml.safe_load(
            (checkpoint_dir / "checkpoint_config.yaml").read_text()
        )
        optimizer_class = checkpoint_config["optimizer"]["class"]
        scheduler_class = checkpoint_config["lr_scheduler"]["class"]
        scheduler_last_epoch = int(scheduler_state.get("last_epoch", -1))
        scheduler_last_lr = [
            float(value) for value in scheduler_state.get("_last_lr", [])
        ]
    else:
        raise FileNotFoundError(
            "Checkpoint lacks a coherent optimizer/LR scheduler state layout: "
            f"{checkpoint_dir}"
        )
    return {
        "optimizer_class": optimizer_class,
        "optimizer_state_entries": len(state.get("state", {})),
        "optimizer_group_lrs": [
            float(group["lr"]) for group in state.get("param_groups", [])
        ],
        "lr_scheduler_class": scheduler_class,
        "lr_scheduler_last_epoch": scheduler_last_epoch,
        "lr_scheduler_last_lr": scheduler_last_lr,
    }


def checkpoint_training_state_layout(path: Path) -> str | None:
    """Return the loadable optimizer/LR pair; never combine incompatible halves."""
    path = Path(path)
    legacy = all((path / name).is_file() for name in ("optimizer.pkl", "lr_scheduler.pkl"))
    native = all((path / name).is_file() for name in ("optimizer.bin", "scheduler.bin"))
    if legacy:
        return "legacy"
    if native:
        return "native"
    return None


def validate_checkpoint_ema_metadata(
    path: Path,
    *,
    expected_sigma_rels: list[float] | tuple[float, ...],
    expected_burn_in: int,
) -> dict[str, object]:
    """Require the checkpoint EMA definition to equal the continuation config."""
    path = Path(path)
    config_path = path / "checkpoint_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint metadata is not a mapping: {config_path}")
    actual_sigma_rels = [float(value) for value in config.get("ema_sigma_rels") or []]
    requested_sigma_rels = [float(value) for value in expected_sigma_rels]
    if actual_sigma_rels != requested_sigma_rels:
        raise ValueError(
            "Checkpoint EMA sigma profiles do not match continuation config: "
            f"checkpoint={actual_sigma_rels}, expected={requested_sigma_rels}"
        )
    actual_burn_in = int(config.get("ema_burn_in", -1))
    if actual_burn_in != int(expected_burn_in):
        raise ValueError(
            "Checkpoint EMA burn-in does not match continuation config: "
            f"checkpoint={actual_burn_in}, expected={int(expected_burn_in)}"
        )
    return {
        "ema_sigma_rels": actual_sigma_rels,
        "ema_burn_in": actual_burn_in,
    }


def checkpoint_missing_files(path: Path) -> tuple[str, ...]:
    missing = [name for name in REQUIRED_RESUME_FILES if not (path / name).is_file()]
    if checkpoint_training_state_layout(path) is None:
        missing.append("coherent optimizer/LR scheduler state")
    for label, alternatives in REQUIRED_RESUME_ALTERNATIVES.items():
        if not any((path / name).is_file() for name in alternatives):
            missing.append(label)
    if not any((path / name).is_file() for name in MODEL_WEIGHT_FILES):
        missing.append("model weights")
    checkpoint_config_path = path / "checkpoint_config.yaml"
    if checkpoint_config_path.is_file():
        try:
            checkpoint_config = yaml.safe_load(checkpoint_config_path.read_text())
            sigma_rels = checkpoint_config.get("ema_sigma_rels") or []
        except Exception:
            missing.append("valid checkpoint_config.yaml")
            sigma_rels = []
        if sigma_rels:
            suffix_sets = []
            for profile in range(len(sigma_rels)):
                suffix_sets.append(
                    {
                        candidate.name.split(".", 1)[1]
                        for candidate in (path / "ema").glob(f"{profile}.*.pt")
                    }
                )
            if not suffix_sets or not set.intersection(*suffix_sets):
                missing.append("aligned EMA profile snapshots")
    return tuple(missing)


def checkpoint_is_complete(path: Path) -> bool:
    return path.is_dir() and not checkpoint_missing_files(path)


def _load_torch_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_loadable_checkpoint_state(path: Path) -> dict[str, object]:
    """Actually deserialize and bind the saved model and training state."""
    path = Path(path)
    try:
        model, noise_scheduler, optimizer, lr_scheduler, _augmentations = (
            load_checkpoint_preserving_class(str(path))
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model/optimizer/scheduler state from checkpoint {path}: {exc}"
        ) from exc

    scaler_path = next(
        (
            path / name
            for name in REQUIRED_RESUME_ALTERNATIVES["gradient scaler"]
            if (path / name).is_file()
        ),
        None,
    )
    if scaler_path is None:
        raise FileNotFoundError(f"Missing gradient scaler state under {path}")
    try:
        scaler_state = _load_torch_checkpoint(scaler_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load gradient scaler state {scaler_path}: {exc}"
        ) from exc
    if not isinstance(scaler_state, dict) or not scaler_state:
        raise ValueError(f"Gradient scaler state is empty or invalid: {scaler_path}")

    return {
        "model_class": type(model).__name__,
        "noise_scheduler_class": type(noise_scheduler).__name__,
        "optimizer_class": type(optimizer).__name__,
        "optimizer_state_entries": len(getattr(optimizer, "state", {})),
        "lr_scheduler_class": type(lr_scheduler).__name__,
        "lr_scheduler_last_epoch": int(getattr(lr_scheduler, "last_epoch", -1)),
        "scaler_path": str(scaler_path),
        "scaler_state_keys": len(scaler_state),
    }


def validate_scientific_checkpoint(
    path: Path,
    *,
    optimizer_steps_per_epoch: int,
    microbatches_per_optimizer_step: int,
    expected_ema_step: int,
    expected_ema_sigma_rels: list[float] | tuple[float, ...],
    expected_ema_burn_in: int,
) -> dict[str, object]:
    """Validate full Accelerate state plus every exact post-hoc EMA profile."""
    import torch

    path = Path(path)
    missing = checkpoint_missing_files(path)
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {path}: {missing}")
    epoch = checkpoint_epoch(path)
    if epoch is None:
        raise ValueError(f"Invalid checkpoint name: {path}")
    ema_metadata = validate_checkpoint_ema_metadata(
        path,
        expected_sigma_rels=expected_ema_sigma_rels,
        expected_burn_in=expected_ema_burn_in,
    )
    sigma_rels = ema_metadata["ema_sigma_rels"]
    burn_in = int(ema_metadata["ema_burn_in"])
    updates = (epoch + 1) * int(optimizer_steps_per_epoch)
    accumulation = int(microbatches_per_optimizer_step)
    if accumulation <= 0:
        raise ValueError("microbatches_per_optimizer_step must be positive")
    microbatches = updates * accumulation
    ema_step = int(expected_ema_step)
    calculated_ema_step = microbatches - burn_in
    if ema_step != calculated_ema_step:
        raise ValueError(
            "Explicit EMA step disagrees with optimizer/microbatch clocks: "
            f"explicit={ema_step}, calculated={calculated_ema_step}"
        )
    snapshots = []
    for profile in range(len(sigma_rels)):
        snapshot = path / "ema" / f"{profile}.{ema_step}.pt"
        if not snapshot.is_file():
            raise FileNotFoundError(f"Missing exact EMA snapshot: {snapshot}")
        try:
            state = torch.load(snapshot, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(snapshot, map_location="cpu")
        state = normalize_posthoc_ema_checkpoint_state(
            state,
            expected_step=ema_step,
        )
        if int(state["step"].item()) != ema_step or not bool(state["initted"].item()):
            raise ValueError(f"Invalid EMA step/init state: {snapshot}")
        if not any(str(key).startswith("ema_model.") for key in state):
            raise ValueError(f"EMA snapshot lacks model weights: {snapshot}")
        snapshots.append(str(snapshot))
    training_state = validate_loadable_checkpoint_state(path)
    return {
        "checkpoint": str(path.resolve()),
        "absolute_updates": updates,
        "absolute_microbatches": microbatches,
        "microbatches_per_optimizer_step": accumulation,
        "ema_sigma_rels": sigma_rels,
        "ema_burn_in": burn_in,
        "ema_step": ema_step,
        "ema_snapshots": snapshots,
        "training_state": training_state,
    }


def latest_checkpoint(checkpoint_dir: Path) -> tuple[Path, int]:
    all_candidates = [
        (path, epoch)
        for path in checkpoint_dir.glob("checkpoint-epoch-*")
        if path.is_dir() and (epoch := checkpoint_epoch(path)) is not None
    ]
    candidates = [
        (path, epoch)
        for path, epoch in all_candidates
        if checkpoint_is_complete(path)
    ]
    if not candidates:
        partial = sorted(path.name for path, _epoch in all_candidates)
        detail = f"; incomplete candidates: {partial}" if partial else ""
        raise FileNotFoundError(
            f"No complete checkpoint-epoch-* directories under {checkpoint_dir}{detail}"
        )
    return max(candidates, key=lambda item: item[1])


def validate_resume_target(
    checkpoint_dir: Path,
    target_checkpoint: Path,
    *,
    minimum_checkpoint: Path | None = None,
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
    if target_checkpoint.exists() and not checkpoint_is_complete(target_checkpoint):
        raise FileExistsError(
            "Refusing to overwrite malformed exact target checkpoint: "
            f"{target_checkpoint}; missing={checkpoint_missing_files(target_checkpoint)}"
        )

    current, current_epoch = latest_checkpoint(checkpoint_dir)
    if minimum_checkpoint is not None:
        minimum_checkpoint = minimum_checkpoint.expanduser()
        minimum_epoch = checkpoint_epoch(minimum_checkpoint)
        if minimum_epoch is None:
            raise ValueError(
                f"Invalid required stage-start checkpoint name: {minimum_checkpoint}"
            )
        if minimum_checkpoint.parent.resolve() != checkpoint_dir:
            raise ValueError(
                f"Required stage-start checkpoint {minimum_checkpoint} is not under "
                f"checkpoint directory {checkpoint_dir}"
            )
        if current_epoch < minimum_epoch:
            raise ValueError(
                f"Latest clean checkpoint epoch {current_epoch} is behind required stage start "
                f"epoch {minimum_epoch}"
            )
        if not checkpoint_is_complete(minimum_checkpoint):
            raise FileNotFoundError(
                "Required exact stage-start checkpoint is missing or incomplete: "
                f"{minimum_checkpoint}"
            )
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


def bound_start_epoch(bound: inspect.BoundArguments) -> int:
    """Read the resume epoch from either supported cosmodiff training API."""
    if "start_epoch" in bound.arguments:
        return int(bound.arguments["start_epoch"])

    resume_from_checkpoint = bound.arguments.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        resume_epoch = checkpoint_epoch(Path(resume_from_checkpoint))
        if resume_epoch is None:
            raise ValueError(
                "Could not derive start epoch from resume checkpoint: "
                f"{resume_from_checkpoint}"
            )
        return resume_epoch + 1

    raise RuntimeError(
        "cosmodiff.optim.train must expose start_epoch or resume_from_checkpoint"
    )


def install_exact_target_adapter(
    optim,
    *,
    expected_start_epoch: int,
    target_epoch: int,
    restored_ema: bool = False,
) -> str:
    original_train = optim.train
    semantics = detect_epoch_semantics(original_train)

    def train_to_exact_target(*args, **kwargs):
        signature = inspect.signature(original_train)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if "num_epochs" not in bound.arguments:
            raise RuntimeError("cosmodiff.optim.train must expose num_epochs")
        start_epoch = bound_start_epoch(bound)
        if start_epoch != expected_start_epoch:
            raise RuntimeError(
                f"External trainer selected start epoch {start_epoch}, expected "
                f"{expected_start_epoch} from the clean checkpoint directory"
            )
        runtime_argument = epoch_argument(start_epoch, target_epoch, semantics)
        bound.arguments["num_epochs"] = runtime_argument
        if restored_ema:
            if "ema_burn_in" not in bound.arguments:
                raise RuntimeError(
                    "Restored post-hoc EMA requires an ema_burn_in training argument"
                )
            bound.arguments["ema_burn_in"] = 0
        print(
            "Exact-target epoch adapter: "
            f"semantics={semantics} start_epoch={start_epoch} "
            f"target_epoch={target_epoch} num_epochs_argument={runtime_argument}",
            flush=True,
        )
        return original_train(*bound.args, **bound.kwargs)

    optim.train = train_to_exact_target
    return semantics


def restore_optimizer_and_lr_scheduler(model, checkpoint_dir: Path):
    """Rebuild training objects; Accelerate then restores their exact states."""
    layout = checkpoint_training_state_layout(checkpoint_dir)
    if layout is None:
        raise FileNotFoundError(
            "Checkpoint lacks a coherent optimizer/LR scheduler state layout: "
            f"{checkpoint_dir}"
        )
    optimizer_path = checkpoint_dir / "optimizer.pkl"
    scheduler_path = checkpoint_dir / "lr_scheduler.pkl"
    if layout == "native":
        checkpoint_config = yaml.safe_load(
            (checkpoint_dir / "checkpoint_config.yaml").read_text()
        )
        optimizer_name = checkpoint_config["optimizer"]["class"]
        optimizer_module, optimizer_class = optimizer_name.rsplit(".", 1)
        optimizer_cls = getattr(importlib.import_module(optimizer_module), optimizer_class)
        optimizer = optimizer_cls(model.parameters())
        scheduler_name = checkpoint_config["lr_scheduler"]["class"]
        scheduler_module, scheduler_class = scheduler_name.rsplit(".", 1)
        scheduler_cls = getattr(importlib.import_module(scheduler_module), scheduler_class)
        lr_scheduler = scheduler_cls(
            optimizer,
            **checkpoint_config["lr_scheduler"].get("kwargs", {}),
        )
        import torch

        try:
            optimizer_state = torch.load(
                checkpoint_dir / "optimizer.bin",
                map_location="cpu",
                weights_only=True,
            )
            scheduler_state = torch.load(
                checkpoint_dir / "scheduler.bin",
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            optimizer_state = torch.load(
                checkpoint_dir / "optimizer.bin", map_location="cpu"
            )
            scheduler_state = torch.load(
                checkpoint_dir / "scheduler.bin", map_location="cpu"
            )
        optimizer.load_state_dict(optimizer_state)
        lr_scheduler.load_state_dict(scheduler_state)
        return optimizer, lr_scheduler

    with optimizer_path.open("rb") as handle:
        saved_optimizer = pickle.load(handle)
    with scheduler_path.open("rb") as handle:
        lr_scheduler = pickle.load(handle)

    optimizer_cls = type(saved_optimizer)
    optimizer_signature = inspect.signature(optimizer_cls.__init__)
    optimizer_parameters = optimizer_signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in optimizer_parameters.values()
    )
    optimizer_kwargs = {
        key: value
        for key, value in saved_optimizer.defaults.items()
        if accepts_kwargs or key in optimizer_parameters
    }
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    optimizer.load_state_dict(saved_optimizer.state_dict())
    if hasattr(lr_scheduler, "optimizer"):
        lr_scheduler.optimizer = optimizer
    return optimizer, lr_scheduler


def load_checkpoint_preserving_class(ckpt_path: str):
    """Restore a class-safe model and the complete saved training state."""
    import diffusers

    checkpoint_dir = Path(ckpt_path)
    config_path = checkpoint_dir / "config.json"
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

    noise_scheduler_path = checkpoint_dir / "noise_scheduler.pkl"
    if noise_scheduler_path.exists():
        with noise_scheduler_path.open("rb") as handle:
            noise_scheduler = pickle.load(handle)
    else:
        checkpoint_config = yaml.safe_load(
            (checkpoint_dir / "checkpoint_config.yaml").read_text()
        )
        scheduler_name = checkpoint_config["noise_scheduler"]["class"]
        scheduler_module, scheduler_class = scheduler_name.rsplit(".", 1)
        scheduler_cls = getattr(importlib.import_module(scheduler_module), scheduler_class)
        noise_scheduler = scheduler_cls.from_pretrained(checkpoint_dir)
    optimizer, lr_scheduler = restore_optimizer_and_lr_scheduler(
        model,
        checkpoint_dir,
    )
    restore_random_states(checkpoint_dir)

    augmentations_path = os.path.join(ckpt_path, "augmentations.pkl")
    if os.path.exists(augmentations_path):
        with open(augmentations_path, "rb") as handle:
            augmentations = pickle.load(handle)
    else:
        augmentations = None

    print(
        f"Class-safe resume loader reconstructed {type(model).__name__} "
        f"and restored optimizer/scheduler state from {ckpt_path}",
        flush=True,
    )
    return model, noise_scheduler, optimizer, lr_scheduler, augmentations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cosmodiff-train", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--minimum-checkpoint", type=Path)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--resume-rng-seed", type=int)
    parser.add_argument("--optimizer-steps-per-epoch", type=int)
    parser.add_argument("--resume-ema-step", type=int)
    parser.add_argument("--target-ema-step", type=int)
    parser.add_argument("--resume-audit", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--code-revision")
    args, extra_args = parser.parse_known_args()

    train_script = args.cosmodiff_train.expanduser().resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Missing cosmodiff training script: {train_script}")
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    train_config = config["train"]
    expected_ema_sigma_rels = [
        float(value) for value in train_config["ema_sigma_rels"]
    ]
    expected_ema_burn_in = int(train_config["ema_burn_in"])

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    target_checkpoint = args.target_checkpoint.expanduser()
    current, current_epoch, target_epoch = validate_resume_target(
        checkpoint_dir,
        target_checkpoint,
        minimum_checkpoint=args.minimum_checkpoint,
    )
    resume_contract_values = (
        args.optimizer_steps_per_epoch,
        args.resume_ema_step,
        args.target_ema_step,
        args.resume_audit,
        args.run_name,
        args.code_revision,
    )
    seed_restart = any(value is not None for value in resume_contract_values)
    if seed_restart and not all(value is not None for value in resume_contract_values):
        raise ValueError(
            "Audited exact resume requires --optimizer-steps-per-epoch, "
            "--resume-ema-step, --target-ema-step, "
            "--resume-audit, --run-name, "
            "and --code-revision together"
        )
    if seed_restart and args.minimum_checkpoint is None:
        raise ValueError("Audited exact resume requires --minimum-checkpoint")
    if current_epoch == target_epoch:
        if args.resume_audit is not None and args.optimizer_steps_per_epoch is None:
            raise ValueError("Audited completed target requires optimizer step count")
        if args.optimizer_steps_per_epoch is not None:
            validate_scientific_checkpoint(
                current,
                optimizer_steps_per_epoch=args.optimizer_steps_per_epoch,
                microbatches_per_optimizer_step=int(
                    train_config.get("gradient_accumulation_steps", 1)
                ),
                expected_ema_step=args.target_ema_step,
                expected_ema_sigma_rels=expected_ema_sigma_rels,
                expected_ema_burn_in=expected_ema_burn_in,
            )
        if args.resume_audit is not None:
            write_completed_noop_audit(
                args.resume_audit.expanduser().resolve(),
                checkpoint=current,
                run_name=args.run_name,
                code_revision=args.code_revision,
            )
        print(f"Exact target already exists; no training needed: {current}", flush=True)
        return

    configured_output_dir = Path(config["io"]["output_dir"]).expanduser().resolve()
    if configured_output_dir != checkpoint_dir:
        raise ValueError(
            f"Config output directory {configured_output_dir} does not match isolated "
            f"checkpoint directory {checkpoint_dir}"
        )

    if args.resume_rng_seed is not None and not seed_restart:
        raise ValueError(
            "--resume-rng-seed is valid only with the complete audited-resume contract"
        )
    effective_resume_seed = resume_seed_for_checkpoint(
        args.resume_rng_seed,
        current,
        args.minimum_checkpoint,
    )

    from cosmodiff import optim, utils

    # The external training entry point imports cosmodiff.optim in this same
    # process. Its resume path therefore calls this exact loader object.
    utils.load_checkpoint = load_checkpoint_preserving_class
    install_constant_label_adapter(utils)
    install_exact_checkpoint_finder(
        utils,
        checkpoint_dir=checkpoint_dir,
        checkpoint=current,
    )
    if seed_restart:
        from accelerate import Accelerator
        import ema_pytorch

        audit_path = args.resume_audit.expanduser().resolve()
        if audit_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing seed-restart audit: {audit_path}"
            )
        minimum_epoch = checkpoint_epoch(args.minimum_checkpoint.expanduser())
        if minimum_epoch is None:
            raise ValueError(
                f"Invalid required stage-start checkpoint: {args.minimum_checkpoint}"
            )
        context = build_seed_restart_context(
            config,
            checkpoint_epoch=current_epoch,
            optimizer_steps_per_epoch=args.optimizer_steps_per_epoch,
            resume_ema_step=(
                ema_step_for_current_checkpoint(
                    stage_start_ema_step=int(args.resume_ema_step),
                    stage_start_epoch=minimum_epoch,
                    current_epoch=current_epoch,
                    optimizer_steps_per_epoch=int(args.optimizer_steps_per_epoch),
                    microbatches_per_optimizer_step=int(
                        train_config.get("gradient_accumulation_steps", 1)
                    ),
                )
            ),
            resume_seed=effective_resume_seed,
            run_name=args.run_name,
        )
        config_bytes = Path(args.config).read_bytes()
        context.update(
            {
                "code_revision": str(args.code_revision),
                "requested_resume_seed": args.resume_rng_seed,
                "seed_origin_checkpoint": (
                    None
                    if args.minimum_checkpoint is None
                    else str(args.minimum_checkpoint.expanduser().resolve())
                ),
                "config_path": str(Path(args.config).expanduser().resolve()),
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "target_checkpoint": str(target_checkpoint.expanduser().resolve()),
                "checkpoint_training_state": checkpoint_optimizer_summary(current),
            }
        )
        install_seed_restart_accelerator_hooks(
            Accelerator,
            checkpoint=current,
            resume_seed=effective_resume_seed,
            source_updates=int(context["source_updates"]),
            source_microbatches=int(context["source_microbatches"]),
            audit_path=audit_path,
            audit_context=context,
        )
        install_seed_restart_ema_factory(
            ema_pytorch,
            checkpoint=current,
            expected_step=int(context["expected_ema_step"]),
            expected_sigma_rels=context["ema_sigma_rels"],
            expected_burn_in=int(context["original_ema_burn_in"]),
            audit_path=audit_path,
        )
    install_exact_target_adapter(
        optim,
        expected_start_epoch=current_epoch + 1,
        target_epoch=target_epoch,
        restored_ema=seed_restart,
    )
    sys.argv = [str(train_script), "--config", args.config, *extra_args]
    runpy.run_path(str(train_script), run_name="__main__")
    if not checkpoint_is_complete(target_checkpoint):
        raise RuntimeError(
            "External trainer exited without a complete exact target checkpoint: "
            f"{target_checkpoint}; missing={checkpoint_missing_files(target_checkpoint)}"
        )
    if seed_restart:
        target_state = validate_scientific_checkpoint(
            target_checkpoint,
            optimizer_steps_per_epoch=args.optimizer_steps_per_epoch,
            microbatches_per_optimizer_step=int(
                context["microbatches_per_optimizer_step"]
            ),
            expected_ema_step=args.target_ema_step,
            expected_ema_sigma_rels=expected_ema_sigma_rels,
            expected_ema_burn_in=expected_ema_burn_in,
        )
        audit = json.loads(audit_path.read_text())
        audit["target_checkpoint_state"] = target_state
        _write_json_atomic(audit_path, audit)
        if "first_resumed_loss" not in audit or "ema_restore" not in audit:
            raise RuntimeError(
                f"Seed-restart audit is incomplete after training: {audit_path}"
            )


if __name__ == "__main__":
    main()
