"""Pure NumPy input transforms for frozen cosmology probes."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np


Transform = Callable[[np.ndarray], tuple[np.ndarray, dict[str, float]]]


def _images(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(f"Expected (N,1,H,W), got {array.shape}")
    if array.shape[-2] != array.shape[-1]:
        raise ValueError("Probe transforms require square images")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("Probe transforms require floating-point images")
    if not np.isfinite(array).all():
        raise ValueError("Probe transform input contains non-finite values")
    return array


def _out_of_range_fraction(array: np.ndarray) -> float:
    return float(np.mean((array < -1.0) | (array > 1.0)))


def _finish(output: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    array = np.ascontiguousarray(output)
    return array, {"out_of_range_fraction": _out_of_range_fraction(array)}


def identity_transform() -> Transform:
    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        array = _images(images)
        return array, {"out_of_range_fraction": _out_of_range_fraction(array)}

    return transform


def rot90_transform(k: int) -> Transform:
    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return _finish(np.rot90(_images(images), int(k) % 4, axes=(-2, -1)))

    return transform


def flip_transform(axis: int) -> Transform:
    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return _finish(np.flip(_images(images), axis=int(axis)))

    return transform


def dihedral_transform(element: int) -> Transform:
    element = int(element)
    if element not in range(8):
        raise ValueError("dihedral element must lie in 0..7")
    rotation = element % 4

    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        output = np.rot90(_images(images), rotation, axes=(-2, -1))
        if element >= 4:
            output = np.flip(output, axis=-1)
        return _finish(output)

    return transform


def roll_transform(dx: int, dy: int) -> Transform:
    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        output = np.roll(
            _images(images),
            shift=(int(dy), int(dx)),
            axis=(-2, -1),
        )
        return _finish(output)

    return transform


def compose_transforms(*transforms: Transform) -> Transform:
    if not transforms:
        raise ValueError("At least one transform is required")

    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        output = images
        diagnostics: dict[str, float] = {}
        for operation in transforms:
            output, diagnostics = operation(output)
        return output, diagnostics

    return transform


def _frequency_grid(height: int, width: int) -> np.ndarray:
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.rfftfreq(width) * width
    kkx, kky = np.meshgrid(kx, ky)
    return np.sqrt(kkx**2 + kky**2)


def _lowpass_response(
    k_radius: np.ndarray,
    k_cut: float,
    window: str,
    k_nyquist: float,
) -> np.ndarray:
    if k_cut >= k_nyquist:
        return np.ones_like(k_radius, dtype=np.float64)
    if window == "sharp":
        return (k_radius <= k_cut).astype(np.float64)
    if window != "hann":
        raise ValueError(f"Unknown Fourier window: {window}")

    lower = max(0.0, k_cut - 2.0)
    upper = k_cut + 2.0
    response = np.ones_like(k_radius, dtype=np.float64)
    response[k_radius >= upper] = 0.0
    transition = (k_radius > lower) & (k_radius < upper)
    phase = (k_radius[transition] - lower) / max(upper - lower, 1.0e-12)
    response[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return response


def _apply_response(images: np.ndarray, response: np.ndarray) -> np.ndarray:
    array = _images(images).astype(np.float32, copy=False)
    expected_shape = (array.shape[-2], array.shape[-1] // 2 + 1)
    if response.shape != expected_shape:
        raise ValueError(
            f"Fourier response has shape {response.shape}; expected {expected_shape}"
        )
    spectrum = np.fft.rfft2(array, axes=(-2, -1))
    output = np.fft.irfft2(
        spectrum * response[None, None],
        s=array.shape[-2:],
        axes=(-2, -1),
    )
    return output.astype(np.float32, copy=False)


def spectral_filter_transform(kind: str, k_cut: float, window: str) -> Transform:
    kind = str(kind)
    window = str(window)
    k_cut = float(k_cut)
    if kind not in {"lowpass", "highpass"}:
        raise ValueError(f"Unknown spectral filter kind: {kind}")
    if window not in {"sharp", "hann"}:
        raise ValueError(f"Unknown Fourier window: {window}")
    if not np.isfinite(k_cut) or k_cut < 0.0:
        raise ValueError("k_cut must be finite and non-negative")

    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        array = _images(images)
        height, width = array.shape[-2:]
        k_nyquist = min(height, width) / 2.0
        lowpass = _lowpass_response(
            _frequency_grid(height, width),
            k_cut,
            window,
            k_nyquist,
        )
        response = lowpass if kind == "lowpass" else 1.0 - lowpass
        output = _apply_response(array, response)
        output, diagnostics = _finish(output)
        diagnostics.update(
            {
                "k_cut": k_cut,
                "k_cut_over_knyq": k_cut / k_nyquist,
            }
        )
        return output, diagnostics

    return transform


def fft_roundtrip_transform() -> Transform:
    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        array = _images(images)
        response = np.ones(
            (array.shape[-2], array.shape[-1] // 2 + 1),
            dtype=np.float64,
        )
        return _finish(_apply_response(array, response))

    return transform


def transfer_transform(
    k_bins: np.ndarray,
    transfer_values: np.ndarray,
) -> Transform:
    centers = np.asarray(k_bins, dtype=np.float64)
    values = np.asarray(transfer_values, dtype=np.float64)
    if centers.ndim != 1 or values.ndim != 1 or centers.shape != values.shape:
        raise ValueError("k_bins and transfer_values must be equal-length vectors")
    if len(centers) < 2:
        raise ValueError("At least two transfer bins are required")
    if not np.isfinite(centers).all() or not np.isfinite(values).all():
        raise ValueError("Transfer bins and values must be finite")
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError("k_bins must be strictly increasing")
    if np.any(values < 0.0):
        raise ValueError("transfer_values must be non-negative")

    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    def transform(images: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        array = _images(images)
        k_radius = _frequency_grid(array.shape[-2], array.shape[-1])
        indices = np.searchsorted(edges, k_radius, side="right") - 1
        indices = np.clip(indices, 0, len(values) - 1)
        response = values[indices]
        response[k_radius == 0.0] = 1.0
        return _finish(_apply_response(array, response))

    return transform


def get_transform(
    name: str,
    *,
    transfer_k: np.ndarray | None = None,
    transfer_values: np.ndarray | None = None,
) -> Transform:
    """Resolve one registered transform by its stable manifest name."""
    if name == "identity":
        return identity_transform()
    if name == "flip_h":
        return flip_transform(-1)
    if name == "flip_v":
        return flip_transform(-2)
    if name == "fft_roundtrip_null":
        return fft_roundtrip_transform()
    if name == "transfer_Tk":
        if transfer_k is None or transfer_values is None:
            raise ValueError("transfer_Tk requires transfer_k and transfer_values")
        return transfer_transform(transfer_k, transfer_values)

    rotation_match = re.fullmatch(r"rot90_k([123])", name)
    if rotation_match:
        return rot90_transform(int(rotation_match.group(1)))

    dihedral_match = re.fullmatch(r"dihedral_g([0-7])", name)
    if dihedral_match:
        return dihedral_transform(int(dihedral_match.group(1)))

    roll_match = re.fullmatch(r"roll_dx(-?\d+)_dy(-?\d+)", name)
    if roll_match:
        return roll_transform(int(roll_match.group(1)), int(roll_match.group(2)))

    filter_match = re.fullmatch(
        r"(lowpass|highpass)_kcut([0-9]+(?:\.[0-9]+)?)_(sharp|hann)",
        name,
    )
    if filter_match:
        return spectral_filter_transform(
            filter_match.group(1),
            float(filter_match.group(2)),
            filter_match.group(3),
        )

    raise KeyError(f"Unknown probe transform: {name}")
