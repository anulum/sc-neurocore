# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GPU/CPU dual-path array module for SC-NeuroCore

from __future__ import annotations

import warnings
from typing import Any

"""
GPU/CPU dual-path array module for SC-NeuroCore.

Uses CuPy when a CUDA GPU is available, falls back to NumPy transparently.
Import ``xp`` as your array library — all standard NumPy operations work
on either backend.

Usage::

    from sc_neurocore.accel.gpu_backend import xp, HAS_CUPY, to_device, to_host

    a = xp.random.random((1024, 1024), dtype=xp.float32)
    packed = gpu_pack_bitstream(bits)          # works on GPU or CPU
    result = gpu_vec_mac(weights, inputs)      # same API, accelerated
"""


import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
try:
    import cupy as cp

    # Verify a device is actually reachable
    cp.cuda.Device(0).compute_capability  # pragma: no cover
    HAS_CUPY = True  # pragma: no cover
    xp = cp  # pragma: no cover
except (ImportError, RuntimeError, AttributeError):
    HAS_CUPY = False
    xp = np

_GPU_WARNED = False


def _warn_cpu_fallback() -> None:
    global _GPU_WARNED  # noqa: PLW0603
    if not HAS_CUPY and not _GPU_WARNED:
        _GPU_WARNED = True
        warnings.warn(
            "CuPy not available; GPU functions are running on CPU via NumPy "
            "fallback. Install cupy-cuda12x for GPU acceleration.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Transfer helpers
# ---------------------------------------------------------------------------


def to_device(arr: np.ndarray[Any, Any]) -> xp.ndarray:  # type: ignore
    """Move a NumPy array to the active backend (GPU copy or no-op)."""
    if HAS_CUPY:  # pragma: no cover
        return cp.asarray(arr)
    return arr


def to_host(arr) -> np.ndarray[Any, Any]:  # type: ignore
    """Bring an array back to host RAM as a NumPy array."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):  # pragma: no cover
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# GPU-accelerated bitstream primitives
# ---------------------------------------------------------------------------


def gpu_pack_bitstream(bits: xp.ndarray) -> xp.ndarray:  # type: ignore
    """
    Pack uint8 {0,1} array into uint64 words.

    Works on both CuPy and NumPy arrays.

    Args:
        bits: Shape ``(N,)`` or ``(B, N)`` of uint8.

    Returns:
        Packed uint64 array, shape ``(ceil(N/64),)`` or ``(B, ceil(N/64))``.
    """
    _warn_cpu_fallback()
    bits = xp.asarray(bits, dtype=xp.uint8)

    if bits.ndim == 1:
        length = bits.size
        pad = (64 - length % 64) % 64
        if pad:
            bits = xp.concatenate([bits, xp.zeros(pad, dtype=xp.uint8)])
        chunks = bits.reshape(-1, 64)
        powers = xp.uint64(1) << xp.arange(64, dtype=xp.uint64)
        return (chunks.astype(xp.uint64) * powers).sum(axis=1)

    elif bits.ndim == 2:
        B, length = bits.shape
        pad = (64 - length % 64) % 64
        if pad:
            bits = xp.concatenate([bits, xp.zeros((B, pad), dtype=xp.uint8)], axis=1)
        n_words = bits.shape[1] // 64
        chunks = bits.reshape(B, n_words, 64)
        powers = xp.uint64(1) << xp.arange(64, dtype=xp.uint64)
        return (chunks.astype(xp.uint64) * powers).sum(axis=2)

    raise ValueError(f"Expected 1-D or 2-D, got {bits.ndim}-D")


def gpu_vec_and(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:  # type: ignore
    """Bitwise AND on packed uint64 arrays (SC multiplication)."""
    _warn_cpu_fallback()
    return xp.bitwise_and(a, b)


def gpu_popcount(packed: xp.ndarray) -> xp.ndarray:  # type: ignore
    """
    Vectorised SWAR popcount on uint64 arrays — returns per-element counts.

    On CuPy this runs as a fused GPU kernel; on NumPy it uses the same
    SWAR bit-trick as ``vector_ops.vec_popcount`` but returns an array
    instead of a scalar.
    """
    _warn_cpu_fallback()
    x = packed.astype(xp.uint64).copy()
    m1 = xp.uint64(0x5555555555555555)
    m2 = xp.uint64(0x3333333333333333)
    m4 = xp.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = xp.uint64(0x0101010101010101)

    x -= (x >> xp.uint64(1)) & m1
    x = (x & m2) + ((x >> xp.uint64(2)) & m2)
    x = (x + (x >> xp.uint64(4))) & m4
    return (x * h01) >> xp.uint64(56)


def gpu_vec_mac(
    packed_weights: xp.ndarray,  # type: ignore
    packed_inputs: xp.ndarray,  # type: ignore
) -> xp.ndarray:  # type: ignore
    """
    GPU-accelerated multiply-accumulate for a dense SC layer.

    Args:
        packed_weights: ``(n_neurons, n_inputs, n_words)`` uint64
        packed_inputs:  ``(n_inputs, n_words)`` uint64

    Returns:
        ``(n_neurons,)`` total bit counts (= SC dot products).
    """
    _warn_cpu_fallback()
    # Broadcast AND: (N, I, W) & (1, I, W) -> (N, I, W)
    products = xp.bitwise_and(packed_weights, packed_inputs[None, :, :])

    # Per-element popcount, then sum across inputs and words
    counts = gpu_popcount(products)  # (N, I, W) uint64
    return counts.sum(axis=(1, 2))  # (N,)
