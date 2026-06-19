# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GPU/CPU dual-path array module for SC-NeuroCore

"""CuPy/NumPy dual-path backend for stochastic-computing array kernels.

The module exposes a runtime-switching array namespace plus helpers for moving
arrays between host and device, packing bitstreams, popcounting packed words,
and running stochastic vector operations with a deterministic NumPy fallback.
"""

from __future__ import annotations

import warnings
from typing import Any

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

_GPU_WARNED = False
_GPU_RUNTIME_BROKEN = False


class _BackendProxy:
    """Runtime-switching array namespace proxy."""

    def __getattr__(self, name: str) -> Any:
        backend = cp if _gpu_enabled() else np
        return getattr(backend, name)


def _gpu_enabled() -> bool:
    return HAS_CUPY and not _GPU_RUNTIME_BROKEN


def _mark_gpu_runtime_broken(exc: RuntimeError) -> None:
    global _GPU_RUNTIME_BROKEN  # noqa: PLW0603
    _GPU_RUNTIME_BROKEN = True
    warnings.warn(
        "CuPy is installed but the CUDA runtime/toolchain is not usable; "
        f"falling back to NumPy CPU execution. Original error: {exc}",
        UserWarning,
        stacklevel=3,
    )


xp = _BackendProxy()


def _warn_cpu_fallback() -> None:
    global _GPU_WARNED  # noqa: PLW0603
    if not _gpu_enabled() and not _GPU_WARNED:
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
    if _gpu_enabled():  # pragma: no cover
        try:
            return cp.asarray(arr)
        except RuntimeError as exc:  # pragma: no cover
            _mark_gpu_runtime_broken(exc)
    return arr


def to_host(arr: Any) -> np.ndarray[Any, Any]:
    """Bring an array back to host RAM as a NumPy array."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):  # pragma: no cover
        try:
            result: np.ndarray[Any, Any] = arr.get()
            return result
        except RuntimeError as exc:  # pragma: no cover
            _mark_gpu_runtime_broken(exc)
    out: np.ndarray[Any, Any] = np.asarray(arr)
    return out


def _numpy_pack_bitstream(bits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    bits_np = np.asarray(bits, dtype=np.uint8)

    if bits_np.ndim == 1:
        bits_1d = bits_np
        length = bits_1d.size
        pad = (64 - length % 64) % 64
        if pad:
            bits_1d = np.concatenate([bits_1d, np.zeros(pad, dtype=np.uint8)])
        chunks = bits_1d.reshape(-1, 64)
        powers = np.uint64(1) << np.arange(64, dtype=np.uint64)
        packed_1d: np.ndarray[Any, Any] = (chunks.astype(np.uint64) * powers).sum(
            axis=1, dtype=np.uint64
        )
        return packed_1d

    if bits_np.ndim == 2:
        bits_2d = bits_np
        batch, length = bits_2d.shape
        pad = (64 - length % 64) % 64
        if pad:
            bits_2d = np.concatenate([bits_2d, np.zeros((batch, pad), dtype=np.uint8)], axis=1)
        n_words = bits_2d.shape[1] // 64
        chunks_2d: np.ndarray[Any, Any] = bits_2d.reshape(batch, n_words, 64)
        powers = np.uint64(1) << np.arange(64, dtype=np.uint64)
        packed_2d: np.ndarray[Any, Any] = (chunks_2d.astype(np.uint64) * powers).sum(
            axis=2, dtype=np.uint64
        )
        return packed_2d

    raise ValueError(f"Expected 1-D or 2-D, got {bits_np.ndim}-D")


def _numpy_popcount(packed: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    x = np.asarray(packed, dtype=np.uint64).copy()
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)

    x -= (x >> np.uint64(1)) & m1
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return (x * h01) >> np.uint64(56)


# ---------------------------------------------------------------------------
# GPU-accelerated bitstream primitives
# ---------------------------------------------------------------------------


def gpu_pack_bitstream(bits: xp.ndarray) -> xp.ndarray:  # type: ignore
    """
    Pack uint8 {0,1} array into uint64 words.

    Works on both CuPy and NumPy arrays.

    Args:
        bits: Shape ``(N,)`` or ``(B, N)`` of uint8.

    Returns
    -------
        Packed uint64 array, shape ``(ceil(N/64),)`` or ``(B, ceil(N/64))``.
    """
    if _gpu_enabled():  # pragma: no cover
        try:
            bits = cp.asarray(bits, dtype=cp.uint8)
            if bits.ndim == 1:
                length = bits.size
                pad = (64 - length % 64) % 64
                if pad:
                    bits = cp.concatenate([bits, cp.zeros(pad, dtype=cp.uint8)])
                chunks = bits.reshape(-1, 64)
                powers = cp.uint64(1) << cp.arange(64, dtype=cp.uint64)
                return (chunks.astype(cp.uint64) * powers).sum(axis=1)

            if bits.ndim == 2:
                batch, length = bits.shape
                pad = (64 - length % 64) % 64
                if pad:
                    bits = cp.concatenate(
                        [bits, cp.zeros((batch, pad), dtype=cp.uint8)],
                        axis=1,
                    )
                n_words = bits.shape[1] // 64
                chunks = bits.reshape(batch, n_words, 64)
                powers = cp.uint64(1) << cp.arange(64, dtype=cp.uint64)
                return (chunks.astype(cp.uint64) * powers).sum(axis=2)
        except RuntimeError as exc:  # pragma: no cover
            _mark_gpu_runtime_broken(exc)

    _warn_cpu_fallback()
    return _numpy_pack_bitstream(to_host(bits))


def gpu_vec_and(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:  # type: ignore
    """Bitwise AND on packed uint64 arrays (SC multiplication)."""
    if _gpu_enabled():  # pragma: no cover
        try:
            return cp.bitwise_and(a, b)
        except RuntimeError as exc:  # pragma: no cover
            _mark_gpu_runtime_broken(exc)
    _warn_cpu_fallback()
    return np.bitwise_and(to_host(a), to_host(b))


def gpu_popcount(packed: xp.ndarray) -> xp.ndarray:  # type: ignore
    """
    Vectorised SWAR popcount on uint64 arrays — returns per-element counts.

    On CuPy this runs as a fused GPU kernel; on NumPy it uses the same
    SWAR bit-trick as ``vector_ops.vec_popcount`` but returns an array
    instead of a scalar.
    """
    if _gpu_enabled():  # pragma: no cover
        try:
            x = cp.asarray(packed, dtype=cp.uint64).copy()
            m1 = cp.uint64(0x5555555555555555)
            m2 = cp.uint64(0x3333333333333333)
            m4 = cp.uint64(0x0F0F0F0F0F0F0F0F)
            h01 = cp.uint64(0x0101010101010101)

            x -= (x >> cp.uint64(1)) & m1
            x = (x & m2) + ((x >> cp.uint64(2)) & m2)
            x = (x + (x >> cp.uint64(4))) & m4
            return (x * h01) >> cp.uint64(56)
        except RuntimeError as exc:  # pragma: no cover
            _mark_gpu_runtime_broken(exc)

    _warn_cpu_fallback()
    return _numpy_popcount(to_host(packed))


def gpu_vec_mac(
    packed_weights: xp.ndarray,  # type: ignore
    packed_inputs: xp.ndarray,  # type: ignore
) -> xp.ndarray:  # type: ignore
    """
    GPU-accelerated multiply-accumulate for a dense SC layer.

    Args:
        packed_weights: ``(n_neurons, n_inputs, n_words)`` uint64
        packed_inputs:  ``(n_inputs, n_words)`` uint64

    Returns
    -------
        ``(n_neurons,)`` total bit counts (= SC dot products).
    """
    if _gpu_enabled():  # pragma: no cover
        try:
            products = cp.bitwise_and(packed_weights, packed_inputs[None, :, :])
            counts = gpu_popcount(products)
            return counts.sum(axis=(1, 2))
        except RuntimeError as exc:  # pragma: no cover
            _mark_gpu_runtime_broken(exc)

    _warn_cpu_fallback()
    weights_np = to_host(packed_weights)
    inputs_np = to_host(packed_inputs)
    products = np.bitwise_and(weights_np, inputs_np[None, :, :])
    counts = _numpy_popcount(products)
    return counts.sum(axis=(1, 2))
