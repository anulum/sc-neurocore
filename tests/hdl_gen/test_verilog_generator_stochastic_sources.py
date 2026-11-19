# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog stochastic-source generation tests

"""Standalone LFSR, Sobol, and Halton stochastic-source routing contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator, emit_sources_from_ir


def test_verilog_generator_generate_routes_stochastic_source_layers() -> None:
    """Stochastic source layers emit their standalone source modules alongside top RTL."""
    gen = VerilogGenerator()
    gen.add_layer("StochasticSource", "rng_lfsr", {"source_type": "LFSR", "seed": 0xBEEF})
    gen.add_layer("StochasticSource", "rng_sobol", {"source_type": "Sobol", "seed": 0x0042})
    gen.add_layer("StochasticSource", "rng_halton", {"source_type": "Halton"})

    code = gen.generate()

    assert "module rng_lfsr" in code
    assert "16'hBEEF" in code
    assert "module rng_sobol" in code
    assert "16'h0042" in code
    assert "module rng_halton" in code


def test_emit_sources_from_ir_supports_halton_node_types() -> None:
    """Direct Halton IR node descriptors emit standalone Halton-16 source modules."""
    code = emit_sources_from_ir({"nodes": {"h0": {"type": "halton16"}}})
    assert "module h0" in code
    assert "assign reversed[0] = counter[15];" in code


def test_verilog_generator_standalone_emitter_methods() -> None:
    """Exercise standalone LFSR, Sobol, Halton, quasi-random, Kuramoto, decorrelator, EDT, TMR."""
    gen = VerilogGenerator(module_name="helper_top")
    assert "module sc_lfsr16_source" in gen.emit_lfsr16_source()
    assert "module sc_sobol16_source" in gen.emit_sobol16_source()
    assert "module sc_halton16_source" in gen.emit_halton16_source()
    assert "module sc_sobol16_source" in gen.emit_quasirandom_source("sobol")
    assert "module sc_halton16_source" in gen.emit_quasirandom_source("halton")
    with pytest.raises(ValueError, match="method must be 'sobol' or 'halton'"):
        gen.emit_quasirandom_source("invalid")
    assert "module helper_top" in gen.emit_kuramoto_phase(n_oscillators=2)
    assert "sc_decorrelator" in gen.emit_decorrelator()
    assert "sc_edt_controller" in gen.emit_edt_controller()
    tmr_rtl = gen.emit_tmr_wrapper("sub_core", [("in_x", 4)], [("out_y", 4)])
    assert "module sub_core_tmr" in tmr_rtl


def test_emit_sources_from_ir_object_nodes_and_validation() -> None:
    """Support object-style IR payloads and reject duplicate or malformed source nodes."""
    gen = VerilogGenerator()
    ir_obj = SimpleNamespace(
        nodes=[
            {"type": "lfsr16"},
            SimpleNamespace(type="sobol16", name="obj_sobol", seed=5),
        ]
    )
    emitted = gen.emit_sources_from_ir(ir_obj)
    assert "module sc_stochastic_source_0" in emitted
    assert "module obj_sobol" in emitted

    with pytest.raises(ValueError, match="duplicate stochastic source module name"):
        emit_sources_from_ir(
            [{"type": "lfsr16", "name": "dup"}, {"type": "sobol16", "name": "dup"}]
        )
    with pytest.raises(TypeError, match="mapping or sequence"):
        emit_sources_from_ir(42)
    with pytest.raises(ValueError, match="unsupported stochastic source type"):
        emit_sources_from_ir([{"type": "stochastic_source", "source_type": "mersenne"}])
    with pytest.raises(ValueError, match="missing source_type"):
        emit_sources_from_ir([{"type": "stochastic_source"}])
    with pytest.raises(ValueError, match="mode must be 'sync' or 'async_aer'"):
        gen.generate(mode="invalid_mode")
