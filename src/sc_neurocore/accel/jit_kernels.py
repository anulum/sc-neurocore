# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Packs a uint8 bitstream into uint64 array

"""Numba-accelerated kernels for packed stochastic-computing hot loops.

The module exposes optional JIT implementations for bitstream packing and
packed multiply-accumulate operations while preserving a pure-Python fallback
when Numba is not installed.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar
import numpy as np
import warnings

_F = TypeVar("_F", bound=Callable[..., Any])

# Try to import Numba
try:
    from numba import jit  # pragma: no cover

    HAS_NUMBA = True  # pragma: no cover
except ImportError:
    HAS_NUMBA = False

    # Fallback decorator: returns the original function
    def jit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
        """Return a no-op decorator used when Numba is unavailable."""

        def decorator(func: _F) -> _F:
            return func

        return decorator

    warnings.warn(
        "Numba not found. Using pure Python fallback. Install 'numba' for high performance."
    )


@jit(nopython=True)  # type: ignore[untyped-decorator]
def jit_pack_bits(
    bitstream: np.ndarray[Any, Any], packed_arr: np.ndarray[Any, Any]
) -> None:  # pragma: no cover
    """Pack a uint8 bitstream into a uint64 word array.

    Parameters
    ----------
    bitstream : numpy.ndarray of shape (N,), uint8
        Input bits valued in ``{0, 1}``.
    packed_arr : numpy.ndarray of shape (N // 64,), uint64
        Output array receiving the packed 64-bit words.
    """
    n = bitstream.size
    n_packed = n // 64

    for i in range(n_packed):
        val = np.uint64(0)
        base = i * 64
        for j in range(64):
            if bitstream[base + j] > 0:
                val |= np.uint64(1) << np.uint64(j)
        packed_arr[i] = val


@jit(nopython=True)  # type: ignore[untyped-decorator]
def jit_vec_mac(
    packed_weights: np.ndarray[Any, Any],
    packed_inputs: np.ndarray[Any, Any],
    outputs: np.ndarray[Any, Any],
) -> None:  # pragma: no cover
    """Accumulate a packed bitwise multiply-accumulate (MAC).

    Computes ``outputs[i] = sum(popcount(packed_weights[i] AND packed_inputs))``.

    Parameters
    ----------
    packed_weights : numpy.ndarray of shape (n_neurons, n_inputs, n_words), uint64
        Packed synaptic weight bitstreams.
    packed_inputs : numpy.ndarray of shape (n_inputs, n_words), uint64
        Packed input bitstreams.
    outputs : numpy.ndarray of shape (n_neurons,)
        Output array receiving the accumulated MAC results.
    """
    n_neurons = packed_weights.shape[0]
    n_inputs = packed_weights.shape[1]
    n_words = packed_weights.shape[2]

    for i in range(n_neurons):
        total_bits = 0
        for j in range(n_inputs):
            for k in range(n_words):
                # Bitwise AND = SC Multiplication
                res = packed_weights[i, j, k] & packed_inputs[j, k]

                # Popcount (Hamming Weight)
                # SWAR Algorithm for 64-bit popcount (Safe for Numba nopython mode)
                x = res
                x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
                x = (x & np.uint64(0x3333333333333333)) + (
                    (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
                )
                x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
                x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

                total_bits += x
        outputs[i] = total_bits
