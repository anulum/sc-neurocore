# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHaltonIRResolution from former test_verilog_integration.py

"""Focused suite: TestHaltonIRResolution from former test_verilog_integration.py."""

from __future__ import annotations

from sc_neurocore.hdl_gen.verilog_generator import (
    emit_sources_from_ir,
)


class TestHaltonIRResolution:
    """Test that Halton source type is correctly resolved from IR."""

    def test_halton_source_from_ir(self) -> None:
        """A Halton source in the IR emits a bit-reversal module."""
        ir = {"nodes": [{"type": "halton16", "name": "my_halton"}]}
        code = emit_sources_from_ir(ir)
        assert "module my_halton" in code
        assert "reversed" in code  # Halton uses bit-reversal

    def test_halton_by_source_type(self) -> None:
        """A Halton source type emits a Halton module."""
        ir = {
            "nodes": [{"type": "stochastic_source", "source_type": "halton", "name": "halton_src"}]
        }
        code = emit_sources_from_ir(ir)
        assert "module halton_src" in code

    def test_mixed_sources(self) -> None:
        """Mixed sources each emit their own module."""
        ir = {
            "nodes": [
                {"type": "lfsr16", "name": "lfsr_src"},
                {"type": "sobol16", "name": "sobol_src"},
                {"type": "halton16", "name": "halton_src"},
            ]
        }
        code = emit_sources_from_ir(ir)
        assert "module lfsr_src" in code
        assert "module sobol_src" in code
        assert "module halton_src" in code
