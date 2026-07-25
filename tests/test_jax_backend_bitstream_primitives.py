# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX bitstream packing and popcount contracts

"""Verify JAX bitstream packing, popcount values, and input validation."""

import numpy as np
import pytest

from tests.jax_backend_support import jax_pack_bitstream, jax_popcount, to_host


def test_jax_pack_bitstream_1d() -> None:
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    packed_np = to_host(jax_pack_bitstream(bits))
    assert packed_np.shape == (1,)
    assert packed_np[0] == 13


def test_jax_pack_bitstream_2d_binary_matrix() -> None:
    bits = np.array([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=np.uint8)
    packed_np = to_host(jax_pack_bitstream(bits))
    assert packed_np.shape == (2, 1)
    assert packed_np[0, 0] == 13
    assert packed_np[1, 0] == 10


@pytest.mark.parametrize(
    "bits",
    [
        np.array([1, 0, 2, 1], dtype=np.uint8),
        np.array([1.0, 0.0, 0.25, 1.0], dtype=np.float32),
        np.array([True, False, True, False], dtype=np.bool_),
    ],
)
def test_jax_pack_bitstream_rejects_non_uint8_binary_streams(bits) -> None:
    with pytest.raises(ValueError, match="uint8 binary"):
        jax_pack_bitstream(bits)


def test_jax_pack_bitstream_rejects_empty_streams() -> None:
    with pytest.raises(ValueError, match="empty"):
        jax_pack_bitstream(np.array([], dtype=np.uint8))


def test_jax_popcount() -> None:
    counts_np = to_host(jax_popcount(np.array([13, 255, 0], dtype=np.uint64)))
    assert counts_np[0] == 3
    assert counts_np[1] == 8
    assert counts_np[2] == 0


@pytest.mark.parametrize(
    "packed",
    [
        np.array([13, 255], dtype=np.int64),
        np.array([13.0, 255.0], dtype=np.float64),
        np.array([], dtype=np.uint64),
    ],
)
def test_jax_popcount_rejects_non_uint64_or_empty_inputs(packed) -> None:
    with pytest.raises(ValueError, match="uint64"):
        jax_popcount(packed)
