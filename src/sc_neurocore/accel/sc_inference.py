# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public stochastic-computing inference over pre-packed weights

"""Stable public SC inference entry point over caller-owned packed weight bitstreams.

``sc_forward`` runs a unipolar stochastic matrix-vector product: it encodes the
input probabilities into bitstreams, ANDs each against the caller's pre-packed
weight bitstreams, popcounts the result and divides by the stream length. The
output is an unbiased estimate of ``W @ input_probs``.

The input encoder is the 16-bit LFSR comparator used by the SC-NeuroCore hardware
path (taps 16, 14, 13, 11; ``bit = reg < x_value`` then advance), so the Rust and
NumPy backends produce bit-identical bitstreams for a fixed seed — the result is
identical to the last bit, not merely within stochastic tolerance.

This surface is the stable replacement for the removed
``get_backend`` / ``VectorizedSCLayer(W_packed, encoder, backend=...)`` integration
relied on by the SCPN-CONTROL Petri-net compiler fast path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .vector_ops import pack_bitstream

if TYPE_CHECKING:
    from .backend import Backend

#: LFSR comparison ceiling: ``x_value = round(p * _LFSR_SCALE)``.
_LFSR_SCALE = 1 << 16
#: LFSR register mask (16-bit register).
_LFSR_MASK = (1 << 16) - 1


def _input_seeds(base_seed: int, n_inputs: int) -> npt.NDArray[np.uint32]:
    """Per-input non-zero 16-bit LFSR seeds derived from ``base_seed``."""
    seeds = (np.uint32(base_seed) + np.arange(n_inputs, dtype=np.uint32)) & _LFSR_MASK
    seeds[seeds == 0] = 1
    return seeds.astype(np.uint32)


def _lfsr_encode_bits(
    input_probs: npt.NDArray[np.float64], length: int, base_seed: int
) -> npt.NDArray[np.uint8]:
    """Encode each input probability into a ``length``-bit LFSR comparator stream.

    Returns an ``(n_inputs, length)`` ``uint8`` array of 0/1 bits. The register
    advance and comparison reproduce :class:`engine::encoder::Lfsr16` exactly, so
    the result is bit-identical to the Rust backend.
    """
    n_inputs = input_probs.size
    x_values = np.clip(np.rint(input_probs * _LFSR_SCALE), 0, _LFSR_SCALE).astype(np.int64)
    reg = _input_seeds(base_seed, n_inputs).astype(np.uint32)
    bits = np.empty((n_inputs, length), dtype=np.uint8)
    for tap in range(length):
        bits[:, tap] = (reg < x_values).astype(np.uint8)
        feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & np.uint32(1)
        reg = (((reg << np.uint32(1)) | feedback) & np.uint32(_LFSR_MASK)).astype(np.uint32)
    return bits


def _validate_packed_weights(
    weights_packed: npt.NDArray[np.uint64], input_probs: npt.NDArray[np.float64], length: int
) -> tuple[int, int, int]:
    """Validate shapes and return ``(n_out, n_in, n_words)``."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if weights_packed.ndim != 3:
        raise ValueError(
            f"weights_packed must be 3-D (n_out, n_in, n_words), got {weights_packed.ndim}-D"
        )
    n_out, n_in, n_words = (int(dim) for dim in weights_packed.shape)
    expected_words = (length + 63) // 64
    if n_words != expected_words:
        raise ValueError(
            f"weights_packed last axis must be ceil(length / 64) = {expected_words}, got {n_words}"
        )
    if input_probs.ndim != 1 or input_probs.size != n_in:
        raise ValueError(
            f"input_probs must be 1-D of length n_in={n_in}, got shape {input_probs.shape}"
        )
    if np.any(input_probs < 0.0) or np.any(input_probs > 1.0):
        raise ValueError("input_probs must lie in [0, 1]")
    return n_out, n_in, n_words


def sc_forward_numpy(
    weights_packed: npt.ArrayLike,
    input_probs: npt.ArrayLike,
    length: int,
    seed: int = 0xACE1,
) -> npt.NDArray[np.float64]:
    """NumPy reference for :func:`sc_forward` — the bit-true floor.

    Parameters
    ----------
    weights_packed : array_like
        Pre-packed unipolar weight bitstreams, shape ``(n_out, n_in, n_words)``
        ``uint64`` with ``n_words = ceil(length / 64)``.
    input_probs : array_like
        Input probabilities, shape ``(n_in,)`` ``float64`` in ``[0, 1]``.
    length : int
        Bitstream length.
    seed : int, optional
        Base LFSR seed for the input encoder.

    Returns
    -------
    numpy.ndarray
        ``(n_out,)`` ``float64`` AND-then-popcount estimate of
        ``weights @ input_probs`` divided by ``length``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or probabilities lie outside ``[0, 1]``.
    """
    weights = np.ascontiguousarray(weights_packed, dtype=np.uint64)
    probs = np.ascontiguousarray(input_probs, dtype=np.float64).reshape(-1)
    n_out, n_in, _ = _validate_packed_weights(weights, probs, length)

    bits = _lfsr_encode_bits(probs, length, seed)
    input_words = np.stack([pack_bitstream(bits[i]) for i in range(n_in)]).astype(np.uint64)

    masked = weights & input_words[np.newaxis, :, :]
    counts = np.bitwise_count(masked).sum(axis=(1, 2), dtype=np.int64)
    estimate: npt.NDArray[np.float64] = counts.astype(np.float64) / float(length)
    return estimate


def sc_forward(
    weights_packed: npt.ArrayLike,
    input_probs: npt.ArrayLike,
    *,
    length: int,
    backend: str | Backend = "auto",
    seed: int = 0xACE1,
) -> npt.NDArray[np.float64]:
    """Stochastic forward pass over caller-owned packed weight bitstreams.

    Encodes ``input_probs`` into LFSR bitstreams, ANDs them against the pre-packed
    weights and returns the popcount estimate of ``weights @ input_probs``. The
    accelerated and NumPy backends are bit-identical for a fixed ``seed``.

    Parameters
    ----------
    weights_packed : array_like
        ``(n_out, n_in, n_words)`` ``uint64`` packed unipolar weight bitstreams.
    input_probs : array_like
        ``(n_in,)`` ``float64`` input probabilities in ``[0, 1]``.
    length : int
        Bitstream length (keyword-only).
    backend : str or Backend, optional
        ``"auto"`` selects the fastest available backend; a name or a
        :class:`~sc_neurocore.accel.backend.Backend` forces one.
    seed : int, optional
        Base LFSR seed for the input encoder.

    Returns
    -------
    numpy.ndarray
        ``(n_out,)`` ``float64`` estimate of ``weights @ input_probs``.
    """
    from .backend import get_backend

    handle = get_backend(backend) if isinstance(backend, str) else backend
    return handle.sc_forward(weights_packed, input_probs, length=length, seed=seed)


__all__ = ["sc_forward", "sc_forward_numpy"]
