# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vector bitstream validation and performance tests

"""Rank validation and opt-in packing performance contracts."""

import time

import numpy as np
import pytest

from sc_neurocore.accel.vector_ops import pack_bitstream, unpack_bitstream
from tests.accel.vector_ops_support import _perf_enabled
from tests.performance_guard import assert_load_tolerant_throughput


def test_pack_bitstream_rejects_3d_input() -> None:
    """pack_bitstream accepts only 1D or 2D arrays."""
    with pytest.raises(ValueError, match="Expected 1D or 2D array"):
        pack_bitstream(np.zeros((2, 2, 2), dtype=np.uint8))


def test_unpack_bitstream_rejects_3d_packed() -> None:
    """unpack_bitstream accepts only 1D or 2D packed arrays."""
    with pytest.raises(ValueError, match="Expected 1D or 2D packed array"):
        unpack_bitstream(np.zeros((2, 2, 2), dtype=np.uint64), 8)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_vector_ops_perf_pack() -> None:
    """Benchmark packing a large bitstream."""
    bits = np.random.randint(0, 2, size=100_000, dtype=np.uint8)
    start = time.perf_counter()
    _ = pack_bitstream(bits)
    elapsed = time.perf_counter() - start
    assert_load_tolerant_throughput(
        label="vector packing run",
        observed_per_second=1.0 / elapsed,
        strict_minimum_per_second=1.0 / 3.0,
    )
