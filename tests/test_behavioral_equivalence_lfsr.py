# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFSR from former test_behavioral_equivalence.py

"""Focused suite: TestLFSR from former test_behavioral_equivalence.py."""

from __future__ import annotations

from tests.behavioral_equivalence_support import *  # noqa: F403


class TestLFSR:
    """Verify LFSR matches Verilog sc_bitstream_encoder polynomial."""

    def test_default_seed(self):
        lfsr = FixedPointLFSR(seed=0xACE1)
        assert lfsr.reg == 0xACE1

    def test_first_step(self):
        """
        0xACE1 = 1010 1100 1110 0001
        Taps: bit15=1, bit13=1, bit12=0, bit10=1 -> feedback = 1^1^0^1 = 1
        Shift left: 0101 1001 1100 0010 = 0x59C2
        Insert feedback at LSB: 0x59C3
        """
        lfsr = FixedPointLFSR(seed=0xACE1)
        result = lfsr.step()
        assert result == 0x59C3, f"Expected 0x59C3, got {hex(result)}"

    def test_maximal_length(self):
        """A 16-bit maximal-length LFSR should have period 2^16-1 = 65535."""
        lfsr = FixedPointLFSR(seed=0xACE1)
        initial = lfsr.reg
        for i in range(65535):
            lfsr.step()
        assert lfsr.reg == initial, "LFSR did not return to initial state after 2^16-1 steps"

    def test_no_zero_state(self):
        """LFSR should never reach all-zeros during full period."""
        lfsr = FixedPointLFSR(seed=0xACE1)
        for _ in range(65535):
            lfsr.step()
            assert lfsr.reg != 0, "LFSR reached zero state"

    def test_zero_seed_raises(self):
        with pytest.raises(ValueError):
            FixedPointLFSR(seed=0)

    def test_different_seeds_produce_different_sequences(self):
        """Two LFSRs with different seeds must produce different bitstreams."""
        lfsr_a = FixedPointLFSR(seed=0xACE1)
        lfsr_b = FixedPointLFSR(seed=0xBEEF)

        seq_a = [lfsr_a.step() for _ in range(100)]
        seq_b = [lfsr_b.step() for _ in range(100)]

        assert seq_a != seq_b, "Different seeds produced identical sequences"
