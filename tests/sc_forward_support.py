# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_forward.py

from __future__ import annotations

"""Tests for the public ``sc_neurocore.accel.sc_forward`` surface required by SCPN-CONTROL.

Covers the NEU-SCPN.4 acceptance gate: ``sc_forward`` estimates ``W @ probs`` within
stochastic tolerance, and the Rust path and NumPy fallback are bit-identical for a
fixed seed.
"""
import numpy as np
import numpy.testing as npt
import numpy.typing as nptyp
import pytest
from sc_neurocore import BitstreamEncoder
from sc_neurocore.accel import backend as backend_mod
from sc_neurocore.accel import (
    available_backends,
    get_backend,
    sc_forward,
    sc_forward_numpy,
)
from sc_neurocore.accel.backend import NumpyBackend, RustBackend
from sc_neurocore.accel.sc_inference import _lfsr_encode_bits
from sc_neurocore.accel.vector_ops import pack_bitstream

_RUST_AVAILABLE = available_backends()["rust"]


def _pack_weights(
    weights: nptyp.NDArray[np.float64], length: int, seed: int
) -> nptyp.NDArray[np.uint64]:
    """Encode weight probabilities into packed bitstreams (decorrelated from inputs)."""
    n_out, n_in = weights.shape
    bits = _lfsr_encode_bits(
        np.ascontiguousarray(weights, dtype=np.float64).reshape(-1), length, seed
    )
    packed = np.stack([pack_bitstream(bits[k]) for k in range(n_out * n_in)])
    return packed.reshape(n_out, n_in, -1).astype(np.uint64)


__all__ = [
    "np",
    "npt",
    "nptyp",
    "pytest",
    "BitstreamEncoder",
    "backend_mod",
    "available_backends",
    "get_backend",
    "sc_forward",
    "sc_forward_numpy",
    "NumpyBackend",
    "RustBackend",
    "_lfsr_encode_bits",
    "pack_bitstream",
    "_RUST_AVAILABLE",
    "_pack_weights",
]
