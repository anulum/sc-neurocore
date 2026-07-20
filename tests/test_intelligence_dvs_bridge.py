# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS AER bridge contracts

"""Contracts for compiler-generated DVS AER bridges."""

from __future__ import annotations


class TestDVSBridge:
    """Tests for DVS→AER bridge Verilog generation."""

    def test_basic_bridge(self) -> None:
        """Default bridge generates valid Verilog."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge()
        assert "module sc_dvs_aer_bridge" in v
        assert "dvs_valid" in v
        assert "dvs_ready" in v
        assert "aer_req" in v
        assert "aer_ack" in v
        assert "fifo_mem" in v
        assert "endmodule" in v

    def test_custom_widths(self) -> None:
        """Custom address and timestamp widths."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge(addr_width=20, timestamp_width=48)
        assert "[19:0]" in v
        assert "[47:0]" in v

    def test_custom_module_name(self) -> None:
        """Custom module name."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge(module_name="my_dvs_bridge")
        assert "module my_dvs_bridge" in v

    def test_fifo_depth(self) -> None:
        """FIFO depth affects address widths."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge(fifo_depth=128)
        assert "[0:127]" in v  # 128-deep FIFO

    def test_polarity_bit_included(self) -> None:
        """Polarity bit appears in ports."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge(polarity_bit=True)
        assert "dvs_polarity" in v
        assert "aer_polarity" in v

    def test_overflow_flag(self) -> None:
        """FIFO overflow detection present."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge()
        assert "fifo_overflow" in v
        assert "overflow_r" in v
