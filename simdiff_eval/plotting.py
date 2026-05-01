"""Small plotting helpers for evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import batch_power_spectra, field_histogram


def save_histogram_plot(real: np.ndarray, generated: np.ndarray, output: str | Path) -> None:
    """Save real/generated one-point histogram comparison."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    real_h = field_histogram(real)
    gen_h = field_histogram(generated)

    edges = np.asarray(real_h["bin_edges"])
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, real_h["hist"], color="black", label="real")
    ax.plot(centers, gen_h["hist"], color="tab:blue", label="generated")
    ax.set_yscale("log")
    ax.set_xlabel("normalized field value")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_power_ratio_plot(real: np.ndarray, generated: np.ndarray, output: str | Path, nbins: int = 25) -> None:
    """Save generated/real mean power-spectrum ratio."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pk_real, kbins = batch_power_spectra(real, nbins=nbins)
    pk_gen, _ = batch_power_spectra(generated, nbins=nbins)
    ratio = np.nanmean(pk_gen, axis=0) / np.clip(np.nanmean(pk_real, axis=0), 1e-30, None)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kbins, ratio, marker="o", color="tab:blue")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.5)
    ax.set_xlabel("k bin")
    ax.set_ylabel("mean generated P(k) / real P(k)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
