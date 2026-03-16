# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for JAX Backend

import pytest
import numpy as np

pytest.importorskip("jax")

from sc_neurocore.accel.jax_backend import (
    jax_pack_bitstream,
    jax_popcount,
    jax_vec_mac,
    to_host,
)


def test_jax_pack_bitstream_1d():
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    packed = jax_pack_bitstream(bits)
    packed_np = to_host(packed)

    assert packed_np.shape == (1,)
    # 1 + 4 + 8 = 13
    assert packed_np[0] == 13


def test_jax_popcount():
    packed = np.array([13, 255, 0], dtype=np.uint64)
    counts = jax_popcount(packed)
    counts_np = to_host(counts)

    assert counts_np[0] == 3
    assert counts_np[1] == 8
    assert counts_np[2] == 0


def test_jax_vec_mac():
    # 2 neurons, 3 inputs, 1 word each
    weights = np.array([[[3], [1], [0]], [[255], [0], [7]]], dtype=np.uint64)

    inputs = np.array([[3], [3], [3]], dtype=np.uint64)

    # weights & inputs
    # Neuron 0:
    # w0 & in0: 3 & 3 = 3 (pop=2)
    # w1 & in1: 1 & 3 = 1 (pop=1)
    # w2 & in2: 0 & 3 = 0 (pop=0)
    # sum = 3

    # Neuron 1:
    # w0 & in0: 255 & 3 = 3 (pop=2)
    # w1 & in1: 0 & 3 = 0 (pop=0)
    # w2 & in2: 7 & 3 = 3 (pop=2)
    # sum = 4

    out = jax_vec_mac(weights, inputs)
    out_np = to_host(out)

    assert out_np.shape == (2,)
    assert out_np[0] == 3
    assert out_np[1] == 4
