# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Packs a uint8 bitstream (0s and 1s) into uint64 integers

"""Packed-bitstream vector operations for stochastic-computing kernels.

This module packs binary streams into ``uint64`` words and provides deterministic
NumPy implementations of Boolean stochastic-computing primitives, unpacking,
and popcount accumulation for tests, CPU execution, and parity checks.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np


def pack_bitstream(bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Pack a uint8 bitstream into uint64 words for 64-way parallel processing.

    Parameters
    ----------
    bitstream : numpy.ndarray of shape (N,) or (Batch, N), uint8
        Input bits valued in ``{0, 1}``.

    Returns
    -------
    numpy.ndarray of shape (ceil(N / 64),) or (Batch, ceil(N / 64)), uint64
        The packed 64-bit words.
    """
    bitstream = np.asarray(bitstream, dtype=np.uint8)

    if bitstream.ndim == 1:
        # 1D case: single bitstream
        length = bitstream.size
        pad_len = (64 - (length % 64)) % 64
        if pad_len > 0:
            bitstream = np.append(bitstream, np.zeros(pad_len, dtype=np.uint8))

        chunks = bitstream.reshape(-1, 64)
        powers = 1 << np.arange(64, dtype=np.uint64)
        packed: np.ndarray[Any, Any] = (chunks * powers).sum(axis=1, dtype=np.uint64)
        return packed

    elif bitstream.ndim == 2:
        # 2D case: batch of bitstreams
        batch_size, length = bitstream.shape
        pad_len = (64 - (length % 64)) % 64

        if pad_len > 0:
            padding = np.zeros((batch_size, pad_len), dtype=np.uint8)
            bitstream = np.concatenate([bitstream, padding], axis=1)

        # Reshape to (batch, num_chunks, 64)
        num_chunks = bitstream.shape[1] // 64
        chunks: np.ndarray[Any, Any] = bitstream.reshape(batch_size, num_chunks, 64)  # type: ignore[no-redef]

        powers = 1 << np.arange(64, dtype=np.uint64)
        packed_2d: np.ndarray[Any, Any] = (chunks * powers).sum(axis=2, dtype=np.uint64)
        return packed_2d

    else:
        raise ValueError(f"Expected 1D or 2D array, got {bitstream.ndim}D")


def unpack_bitstream(
    packed: np.ndarray[Any, Any],
    original_length: int,
    original_shape: Optional[tuple[Any, ...]] = None,
) -> np.ndarray[Any, Any]:
    """
    Unpacks uint64 array back to uint8 bitstream.

    Args:
        packed: Packed uint64 array (1D or 2D)
        original_length: Total number of bits to extract
        original_shape: Optional tuple for reshaping output (batch, length)

    Returns
    -------
        Unpacked bitstream of shape (original_length,) or original_shape
    """
    if packed.ndim == 1:
        # 1D packed array
        bits = ((packed[:, None] & (1 << np.arange(64, dtype=np.uint64))) > 0).astype(np.uint8)
        unpacked = bits.flatten()
        result: np.ndarray[Any, Any] = unpacked[:original_length]
        return result

    elif packed.ndim == 2:
        # 2D packed array: (batch, num_chunks)
        batch_size, num_chunks = packed.shape
        # Extract bits: (batch, num_chunks, 64)
        bits = ((packed[:, :, None] & (1 << np.arange(64, dtype=np.uint64))) > 0).astype(np.uint8)
        # Reshape to (batch, num_chunks * 64)
        unpacked = bits.reshape(batch_size, -1)

        if original_shape is not None:
            result_2d: np.ndarray[Any, Any] = unpacked[:, : original_shape[1]]
            return result_2d
        else:
            # Assume original_length is per-batch
            per_batch_len = original_length // batch_size
            result_batch: np.ndarray[Any, Any] = unpacked[:, :per_batch_len]
            return result_batch

    else:
        raise ValueError(f"Expected 1D or 2D packed array, got {packed.ndim}D")


def vec_and(a_packed: np.ndarray[Any, Any], b_packed: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Bitwise-AND two packed arrays, realising stochastic multiplication."""
    result: np.ndarray[Any, Any] = np.bitwise_and(a_packed, b_packed)
    return result


def vec_xnor(
    a_packed: np.ndarray[Any, Any], b_packed: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Bitwise XNOR on packed arrays. SC bipolar multiplication: P(A XNOR B) = P(A)*P(B) + (1-P(A))*(1-P(B))."""
    result: np.ndarray[Any, Any] = ~np.bitwise_xor(a_packed, b_packed)
    return result


def vec_not(packed: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Bitwise NOT on packed arrays. SC complement: P(NOT A) = 1 - P(A)."""
    result: np.ndarray[Any, Any] = ~packed
    return result


def vec_mux(
    select_packed: np.ndarray[Any, Any],
    a_packed: np.ndarray[Any, Any],
    b_packed: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Bitwise MUX on packed arrays. SC scaled addition: P(out) = P(sel)*P(A) + (1-P(sel))*P(B).

    When sel is a Bernoulli(0.5) stream, this computes the average (A+B)/2.
    """
    result: np.ndarray[Any, Any] = (select_packed & a_packed) | (~select_packed & b_packed)
    return result


def vec_popcount(packed: np.ndarray[Any, Any]) -> int:
    """Count the total set bits in a packed array, for integration/accumulation."""
    # Using numpy's ability to cast to specialized types or simple lookup?
    # Actually, Python 3.10+ int.bit_count() is fast, but for numpy arrays:
    # We can use a trick or just loop if C-extension isn't available.
    # A generic parallel popcount on uint64 in pure numpy is tricky without looping or lookup tables.
    # However, we can map to python int and sum.

    # For speed in pure python/numpy env without heavy deps:
    # Use binary decomposition for vectorized popcount
    x = packed.copy()
    x -= (x >> 1) & 0x5555555555555555
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x * 0x0101010101010101) >> 56
    return int(np.sum(x))
