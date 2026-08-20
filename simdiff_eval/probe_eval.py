"""Shared loading for frozen-probe evaluation on held-out real maps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_nf_conditional_u128_config import (  # noqa: E402
    N_TRAIN_SIMS,
    image_path,
    load_params,
    params_path,
)
from train_nf_conditional_bias_encoder import (  # noqa: E402
    load_raw_slices,
    preprocess_real_slices,
    select_slice_pairs,
)


def load_heldout_real_slices(
    data_root: str | Path,
    heldout_indices: np.ndarray,
    slices_per_sim: int,
    norm: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load normalized held-out slices in the trainer's historical order."""
    heldout = np.asarray(heldout_indices, dtype=np.int64).reshape(-1)
    slices_per_sim = int(slices_per_sim)
    if heldout.size == 0:
        raise ValueError("heldout_indices is empty")
    if slices_per_sim < 1 or slices_per_sim > 128:
        raise ValueError("slices_per_sim must lie in 1..128")

    pairs = select_slice_pairs(heldout, len(heldout) * slices_per_sim)
    raw = load_raw_slices(image_path(data_root), pairs)
    images = preprocess_real_slices(raw, norm).astype(np.float32, copy=False)
    params = load_params(params_path(data_root), N_TRAIN_SIMS)
    sim_index = pairs[:, 0].astype(np.int64, copy=False)
    z_index = pairs[:, 1].astype(np.int64, copy=False)
    theta_raw = params[sim_index].astype(np.float32, copy=False)
    return images, theta_raw, sim_index, z_index
