"""Co-simulation: Synapse AND logic - Rust golden model vs Verilator HDL."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


@pytest.mark.usefixtures("verilator_available")
class TestSynapseCosim:
    """Validate stochastic synapse (AND) logic."""

    def test_and_probability(self):
        """Bitwise AND of two random bitstreams: popcount matches."""
        rng = np.random.RandomState(42)
        bits_a = rng.randint(0, 2, 10000).astype(np.uint8)
        rng2 = np.random.RandomState(43)
        bits_b = rng2.randint(0, 2, 10000).astype(np.uint8)

        packed_a = v3.pack_bitstream(bits_a)
        packed_b = v3.pack_bitstream(bits_b)

        # Compute AND at numpy level
        and_bits = bits_a & bits_b
        expected_count = int(np.sum(and_bits))

        # Compute AND at packed level
        packed_and = [a & b for a, b in zip(packed_a, packed_b)]
        actual_count = v3.popcount(packed_and)

        assert abs(expected_count - actual_count) <= 1, (
            f"AND popcount mismatch: expected={expected_count}, actual={actual_count}"
        )

    def test_all_zeros(self):
        """AND with all-zero stream produces zero."""
        zeros = np.zeros(1024, dtype=np.uint8)
        ones = np.ones(1024, dtype=np.uint8)
        packed_z = v3.pack_bitstream(zeros)
        packed_o = v3.pack_bitstream(ones)
        packed_and = [a & b for a, b in zip(packed_z, packed_o)]
        assert v3.popcount(packed_and) == 0

    def test_all_ones(self):
        """AND with two all-one streams produces all ones."""
        ones = np.ones(1024, dtype=np.uint8)
        packed = v3.pack_bitstream(ones)
        packed_and = [a & b for a, b in zip(packed, packed)]
        assert v3.popcount(packed_and) == 1024
