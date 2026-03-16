# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Strict blueprint semantics tests for LFSR + bitstream

"""Strict blueprint semantics tests for LFSR + bitstream encoder.

The encoder uses compare-before-advance semantics matching the Verilog RTL
(sc_bitstream_encoder.v): non-blocking assignments mean `bit_out` reads
the LFSR state *before* the advance that happens in the same clock edge.
"""

import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

from sc_neurocore_engine import BitstreamEncoder, Lfsr16


def _lfsr_step(reg: int) -> int:
    feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
    return ((reg << 1) & 0xFFFF) | feedback


class TestLFSRBlueprintSemantics:
    def test_full_cycle_matches_blueprint_formula(self):
        reg = 0xACE1
        v3 = Lfsr16(seed=reg)

        for i in range(65535):
            reg = _lfsr_step(reg)
            assert v3.step() == reg, f"LFSR divergence at step {i}"

    def test_multiple_seeds(self):
        for seed in [0xACE1, 0xBEEF, 0xACE1 + 7, 0xBEEF + 13]:
            reg = seed
            v3 = Lfsr16(seed=seed)
            for i in range(1000):
                reg = _lfsr_step(reg)
                assert v3.step() == reg, f"Seed {seed:#06x} diverged at {i}"


class TestEncoderBlueprintSemantics:
    def test_compare_before_advance_order(self):
        """Encoder reads LFSR state, compares, then advances (Verilog RTL match)."""
        encoder = BitstreamEncoder(data_width=16, seed=0xACE1)
        x_value = 0xACE1
        reg = 0xACE1

        bits = []
        for _ in range(8):
            expected = 1 if reg < x_value else 0
            bits.append(encoder.step(x_value))
            assert bits[-1] == expected
            reg = _lfsr_step(reg)

        # 0xACE1 < 0xACE1 is false → first bit = 0
        assert bits[0] == 0, "Compare-before-advance: reg == x_value → bit = 0"
