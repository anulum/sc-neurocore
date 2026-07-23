# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_primitives.py

from __future__ import annotations

"""Tests for SC fundamental gates: MUX (addition), XNOR (bipolar multiply), NOT (complement)."""
import numpy as np
from sc_neurocore.accel.vector_ops import (
    pack_bitstream,
    vec_and,
    vec_xnor,
    vec_not,
    vec_mux,
    vec_popcount,
)
def _prob(packed, length):
    """Estimate probability from packed bitstream."""
    return vec_popcount(packed) / length
def _bernoulli_packed(p, length, seed):
    """Generate a packed Bernoulli bitstream."""
    rng = np.random.RandomState(seed)
    bits = (rng.random(length) < p).astype(np.uint8)
    return pack_bitstream(bits), length

__all__ = ['np', 'pack_bitstream', 'vec_and', 'vec_xnor', 'vec_not', 'vec_mux', 'vec_popcount', '_prob', '_bernoulli_packed']
