# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVerilogGeneratorNewMethods from former test_verilog_integration.py

"""Focused suite: TestVerilogGeneratorNewMethods from former test_verilog_integration.py."""

from __future__ import annotations

from tests.verilog_integration_support import *  # noqa: F403


class TestVerilogGeneratorNewMethods:
    """Test the new emit methods on VerilogGenerator."""

    def test_emit_halton16_source(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_halton16_source()
        assert "module sc_halton16_source" in code
        assert "reversed" in code

    def test_emit_quasirandom_source_sobol(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_quasirandom_source(method="sobol")
        assert "module sc_sobol16_source" in code
        assert "casez" in code

    def test_emit_quasirandom_source_halton(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_quasirandom_source(method="halton")
        assert "module sc_halton16_source" in code

    def test_emit_decorrelator(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_decorrelator(num_streams=4, stream_width=8)
        assert "sc_decorrelator" in code
        assert "NUM_STREAMS(4)" in code
        assert "STREAM_WIDTH(8)" in code

    def test_emit_edt_controller(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_edt_controller(data_width=16, margin=0x0080, stable_cycles=4)
        assert "sc_edt_controller" in code
        assert "DATA_WIDTH(16)" in code
        assert "MARGIN(16'h0080)" in code
        assert "STABLE_CYCLES(4)" in code

    def test_emit_tmr_wrapper(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_tmr_wrapper(
            module_name="sc_aer_router",
            inputs=[("clk", 1), ("rst_n", 1)],
            outputs=[("packet_out", 32)],
        )
        assert "module sc_aer_router_tmr" in code
        assert "replica_0" in code
        assert "replica_1" in code
        assert "replica_2" in code
        assert "packet_out_tmr_error" in code
