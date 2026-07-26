# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog stochastic-source generation tests

"""Standalone LFSR and Sobol stochastic-source routing contracts."""

from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator


def test_verilog_generator_generate_routes_stochastic_source_layers():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Stochastic source layers should emit their standalone source modules."""
    gen = VerilogGenerator()
    gen.add_layer("StochasticSource", "rng_lfsr", {"source_type": "LFSR", "seed": 0xBEEF})
    gen.add_layer("StochasticSource", "rng_sobol", {"source_type": "Sobol", "seed": 0x0042})

    code = gen.generate()

    assert "module rng_lfsr" in code
    assert "16'hBEEF" in code
    assert "module rng_sobol" in code
    assert "16'h0042" in code
