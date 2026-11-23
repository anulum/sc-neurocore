# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFusionVerilogEmitter from former test_sensor_fusion.py

"""Focused suite: TestFusionVerilogEmitter from former test_sensor_fusion.py."""

from __future__ import annotations

from sc_neurocore.fusion.sensor_fusion import (
    FusionVerilogEmitter,
)


class TestFusionVerilogEmitter:
    """SystemVerilog emitted for sensor fusion."""

    def test_emit_contains_module(self) -> None:
        """The emitted text is a complete fusion module."""
        sv = FusionVerilogEmitter.emit()
        assert "module sc_multimodal_fusion" in sv
        assert "endmodule" in sv

    def test_emit_custom_streams(self) -> None:
        """The stream count and bitstream width reach the parameters."""
        sv = FusionVerilogEmitter.emit(num_streams=6, bitstream_width=32)
        assert "STREAMS      = 6" in sv
        assert "BITSTREAM_W  = 32" in sv

    def test_emit_attention_mode(self) -> None:
        """Attention mode emits the stochastic AND coincidence."""
        sv = FusionVerilogEmitter.emit(use_attention=True)
        assert "SC-AND" in sv or "coincidence" in sv

    def test_emit_or_mode(self) -> None:
        """Without attention the module fuses by OR."""
        sv = FusionVerilogEmitter.emit(use_attention=False)
        assert "OR fusion" in sv

    def test_emit_custom_name(self) -> None:
        """A custom module name names the module."""
        sv = FusionVerilogEmitter.emit(module_name="my_fusion")
        assert "module my_fusion" in sv

    def test_lfsr_decorrelation_present(self) -> None:
        """The module decorrelates with an LFSR."""
        sv = FusionVerilogEmitter.emit()
        assert "lfsr" in sv.lower()
        assert "decorr" in sv
