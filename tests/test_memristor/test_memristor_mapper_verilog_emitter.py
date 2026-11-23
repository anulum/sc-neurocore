# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVerilogEmitter from former test_memristor_mapper.py

"""Focused suite: TestVerilogEmitter from former test_memristor_mapper.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.memristor.memristor_mapper import (
    CompensationStrategy,
    MemristorMapper,
    MemristorTechnology,
    VerilogEmitter,
)


class TestVerilogEmitter:
    """SystemVerilog emitted for memristor crossbars."""

    def test_emit_crossbar_module(self) -> None:
        """A crossbar module is emitted."""
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "module sc_memristor_crossbar" in sv
        assert "endmodule" in sv

    def test_emit_contains_spdx(self) -> None:
        """The module carries its SPDX header."""
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "SPDX-License-Identifier" in sv

    def test_emit_weight_parameters(self) -> None:
        """Each weight becomes a parameter."""
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "W_0_0" in sv
        assert "W_3_3" in sv

    def test_emit_compensation_lut(self) -> None:
        """The compensation table is emitted."""
        mapper = MemristorMapper(compensation=CompensationStrategy.LUT, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "comp_lut" in sv

    def test_emit_no_comp_when_none(self) -> None:
        """Without a table the module says so."""
        mapper = MemristorMapper(compensation=CompensationStrategy.NONE, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "No compensation LUT" in sv

    def test_emit_top_module(self) -> None:
        """A top module instantiates every tile."""
        mapper = MemristorMapper(max_crossbar_size=4, seed=42)
        w = np.random.default_rng(0).random((8, 8))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_top(result)
        assert "module sc_memristor_array" in sv
        assert "tile_0" in sv
        assert "tile_1" in sv

    def test_custom_bit_width(self) -> None:
        """A custom bit width sets the port width."""
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((2, 2))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter(bit_width=32)
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "[31:0]" in sv

    def test_emit_technology_in_header(self) -> None:
        """The header names the technology."""
        mapper = MemristorMapper(technology=MemristorTechnology.PCM, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "pcm" in sv
