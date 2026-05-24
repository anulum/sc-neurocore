# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for JIT kernel contracts

"""Contracts for low-level bit-packing JIT kernels."""

from __future__ import annotations

import numpy as np


def test_jit_pack_bits_encodes_zero_and_full_words() -> None:
    from sc_neurocore.accel.jit_kernels import jit_pack_bits

    packed_zero = np.zeros(1, dtype=np.uint64)
    packed_one = np.zeros(1, dtype=np.uint64)

    jit_pack_bits(np.zeros(64, dtype=np.uint8), packed_zero)
    jit_pack_bits(np.ones(64, dtype=np.uint8), packed_one)

    assert packed_zero[0] == 0
    assert packed_one[0] == np.uint64(0xFFFFFFFFFFFFFFFF)
