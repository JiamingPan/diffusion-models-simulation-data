#!/usr/bin/env python
"""Prepare one N=64 no-augmentation memorization diagnostic config.

This is not a replacement for the generalizability sweep. It isolates the
training-budget question for the smallest data setting:

- same u128 Nick-default-style recipe as nf_generalize_nick_data
- same combined LH/CV and z=0/1/2 source allocation
- N=64 two-dimensional slices
- no data augmentation, so exact slice memorization is not hidden by shifts/flips
- many more epochs so the run gets about 100k optimizer updates instead of ~200
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from prepare_nf_generalize_nick_data_configs import (
    ARCH,
    DATA_SOURCES,
    SLICES_PER_VOLUME,
    ZTHIN,
    allocate_source_counts,
    build_config,
)


SWEEP_NAME = "nf_generalize_n64_memorize"
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{SWEEP_NAME}"
RUN_NAME = "nf_gen_nick_u128_d2p06_noaug_mem100k"
DATASET_TAG = "d2p06_noaug_mem100k"
TARGET_2D = 64
TARGET_UPDATES = 100_000
CHECKPOINT_EVERY_UPDATES = 2_000


def build_memorize_config() -> dict[str, Any]:
    sources = allocate_source_counts(TARGET_2D)
    config = deepcopy(build_config(RUN_NAME, sources))
    steps_per_epoch = max(1, TARGET_2D // int(ARCH["batch_size"]))
    num_epochs = TARGET_UPDATES // steps_per_epoch
    checkpoint_every_n_epochs = max(1, CHECKPOINT_EVERY_UPDATES // steps_per_epoch)

    config["io"]["output_dir"] = f"{CHECKPOINT_ROOT}/{RUN_NAME}_checkpoints"
    config.pop("augmentations", None)
    config["train"]["num_epochs"] = int(num_epochs)
    config["train"]["checkpoint_every_n_epochs"] = int(checkpoint_every_n_epochs)
    config["train"]["verbose"] = True
    config["generate"]["n_samples"] = 512
    return config


def manifest_row() -> dict[str, Any]:
    sources = allocate_source_counts(TARGET_2D)
    steps_per_epoch = max(1, TARGET_2D // int(ARCH["batch_size"]))
    num_epochs = TARGET_UPDATES // steps_per_epoch
    actual_updates = num_epochs * steps_per_epoch
    return {
        "run_name": RUN_NAME,
        "arch": ARCH["arch"],
        "arch_label": ARCH["arch_label"],
        "variant_tag": "nick_default_noaug_mem100k",
        "variant_label": "Nick default N=64, no aug, long training",
        "dataset_tag": DATASET_TAG,
        "dataset_group": "LH+CV z=0,1,2",
        "target_2d": TARGET_2D,
        "actual_2d": TARGET_2D,
        "dataset_size": TARGET_2D,
        "n_train_simulations": sum(int(src["n_samples"]) for src in sources),
        "n_samples_simulations": sum(int(src["n_samples"]) for src in sources),
        "zthin": ZTHIN,
        "slices_per_sim": SLICES_PER_VOLUME,
        "source_counts": [
            {
                "tag": src["tag"],
                "path": src["path"],
                "n_samples": int(src["n_samples"]),
                "n_2d_slices": int(src["n_2d_slices"]),
            }
            for src in sources
        ],
        "epochs": int(num_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "target_updates": TARGET_UPDATES,
        "actual_updates": int(actual_updates),
        "checkpoint_every_updates": CHECKPOINT_EVERY_UPDATES,
        "batch_size": ARCH["batch_size"],
        "config": f"local/{SWEEP_NAME}/configs/{RUN_NAME}.yaml",
        "checkpoint_dir": f"{CHECKPOINT_ROOT}/{RUN_NAME}_checkpoints",
        "sample_path": f"results/{SWEEP_NAME}/samples/{RUN_NAME}_seed{{seed}}_raw_train_full.npz",
        "note": "N=64 memorization diagnostic: same Nick-default u128 recipe, no augmentation, about 100k optimizer updates.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--check-only", action="store_true", help="Validate existing config without writing.")
    parser.add_argument("--print-runs", action="store_true", help="Print run name and exit.")
    return parser.parse_args()


def assert_config(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)

    row = manifest_row()
    checks = {
        "run_name": path.stem == RUN_NAME,
        "io.output_dir": config["io"]["output_dir"] == row["checkpoint_dir"],
        "data.img_path_list": isinstance(config["data"].get("img_path"), list),
        "data.n_samples_list": isinstance(config["data"].get("n_samples"), list),
        "data.target_size": sum(config["data"]["n_samples"]) * SLICES_PER_VOLUME == TARGET_2D,
        "train.num_epochs": config["train"].get("num_epochs") == row["epochs"],
        "augmentations.absent": "augmentations" not in config,
        "train.checkpoint_every_n_epochs": (
            config["train"].get("checkpoint_every_n_epochs")
            == max(1, CHECKPOINT_EVERY_UPDATES // row["steps_per_epoch"])
        ),
        "model.arch": config["model"]["kwargs"].get("block_out_channels") == ARCH["block_out_channels"],
        "noise.prediction_type": config["noise_scheduler"]["kwargs"].get("prediction_type") == "v_prediction",
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{path} failed checks: {', '.join(failed)}")


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_dir = project_dir / "local" / SWEEP_NAME / "configs"
    manifest_path = project_dir / "local" / SWEEP_NAME / "manifest.json"

    if args.print_runs:
        print(RUN_NAME)
        return

    config_path = config_dir / f"{RUN_NAME}.yaml"
    if args.check_only:
        assert_config(config_path)
        print(f"Validated {SWEEP_NAME} config.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(build_memorize_config(), f, sort_keys=False)
    print(f"Wrote {config_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump([manifest_row()], f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
