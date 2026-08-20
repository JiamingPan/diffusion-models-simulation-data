#!/usr/bin/env python
"""Validate one exact-checkpoint DiT sample artifact and its provenance."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


AUDIT_KEYS = {
    "scheduler_class",
    "requested_inference_steps",
    "executed_inference_steps",
    "first_timestep",
    "final_timestep",
    "terminal_sigma",
    "terminal_sigma_is_zero",
    "terminal_sigma_verifiable",
}


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    if key not in data.files:
        raise ValueError(f"missing sample provenance key: {key}")
    return data[key].item()


def validate_sample_file(
    path: Path,
    *,
    requested_checkpoint: Path,
    scheduler: str,
    requested_steps: int,
    expected_shape: tuple[int, ...] = (512, 1, 128, 128),
) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing sample file: {path}")
    with np.load(path) as data:
        missing = sorted(AUDIT_KEYS - set(data.files))
        if missing:
            raise ValueError(f"missing scheduler audit metadata: {', '.join(missing)}")
        samples = np.asarray(data["samples"])
        if samples.shape != expected_shape:
            raise ValueError(
                f"sample shape mismatch: expected {expected_shape}, found {samples.shape}"
            )
        if not np.isfinite(samples).all():
            raise ValueError(f"sample tensor contains non-finite values: {path}")

        expected_checkpoint = str(Path(requested_checkpoint))
        requested = str(_scalar(data, "requested_checkpoint"))
        resolved = str(_scalar(data, "resolved_checkpoint"))
        if requested != expected_checkpoint or resolved != expected_checkpoint:
            raise ValueError(
                "checkpoint provenance mismatch: "
                f"expected={expected_checkpoint} requested={requested} resolved={resolved}"
            )
        if str(_scalar(data, "scheduler")) != scheduler:
            raise ValueError("scheduler provenance does not match the requested scheduler")
        if str(_scalar(data, "scheduler_class")) != scheduler:
            raise ValueError("executed scheduler class does not match the requested scheduler")
        if int(_scalar(data, "requested_inference_steps")) != int(requested_steps):
            raise ValueError("requested inference-step metadata mismatch")
        if int(_scalar(data, "executed_inference_steps")) != int(requested_steps):
            raise ValueError("executed inference steps do not equal requested steps")

        return {
            "path": str(path),
            "shape": list(samples.shape),
            "resolved_checkpoint": resolved,
            "scheduler": scheduler,
            "executed_inference_steps": int(
                _scalar(data, "executed_inference_steps")
            ),
            "terminal_sigma": float(_scalar(data, "terminal_sigma")),
            "terminal_sigma_verifiable": bool(
                _scalar(data, "terminal_sigma_verifiable")
            ),
        }


def _sample_digest(path: Path) -> tuple[str, str]:
    with np.load(path) as data:
        samples = np.ascontiguousarray(data["samples"])
        checkpoint = str(_scalar(data, "resolved_checkpoint"))
    return hashlib.sha256(samples.view(np.uint8)).hexdigest(), checkpoint


def reject_cross_checkpoint_duplicates(sample_root: Path, *, sample_label: str) -> None:
    sample_root = Path(sample_root)
    seen: dict[str, tuple[Path, str]] = {}
    for path in sorted(sample_root.glob(f"*_{sample_label}.npz")):
        digest, checkpoint = _sample_digest(path)
        previous = seen.get(digest)
        if previous is not None and previous[1] != checkpoint:
            raise ValueError(
                "byte-identical sample tensors came from different checkpoints: "
                f"{previous[0]} ({previous[1]}) and {path} ({checkpoint})"
            )
        seen[digest] = (path, checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--sample-label", required=True)
    parser.add_argument("--requested-checkpoint", required=True, type=Path)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--requested-steps", required=True, type=int)
    args = parser.parse_args()

    report = validate_sample_file(
        args.sample,
        requested_checkpoint=args.requested_checkpoint,
        scheduler=args.scheduler,
        requested_steps=args.requested_steps,
    )
    reject_cross_checkpoint_duplicates(
        args.sample_root, sample_label=args.sample_label
    )
    print(report)
    print("PASS: sample tensor and scheduler provenance are valid")


if __name__ == "__main__":
    main()
