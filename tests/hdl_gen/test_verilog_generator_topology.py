# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog generator topology tests

"""Module, layer, wire, bus-width, and dense-instance topology contracts."""

from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator


def test_verilog_generator_module_name():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Generated code should contain module name."""
    gen = VerilogGenerator(module_name="my_top")
    code = gen.generate()
    assert "module my_top" in code


def test_verilog_generator_add_layer():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """add_layer should register layers in order."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "layer0", {"n_neurons": 4})
    assert len(gen.layers) == 1
    assert gen.layers[0]["name"] == "layer0"


def test_verilog_generator_single_layer_wiring():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Single Dense layer should connect input_bus to output_bus."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 4})
    code = gen.generate()
    assert ".input_bus(input_bus)" in code
    assert ".output_bus(output_bus)" in code


def test_verilog_generator_two_layers_wires():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Two Dense layers should include inter-layer wire."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 4})
    gen.add_layer("Dense", "dense1", {"n_neurons": 4})
    code = gen.generate()
    assert "wire [3:0] layer_0_to_1;" in code


def test_verilog_generator_derives_declared_bus_widths():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Sync generator should not force every top-level bus to eight bits."""
    gen = VerilogGenerator(module_name="wide_top", bus_width=16)
    gen.add_layer("Dense", "dense0", {"n_neurons": 12, "output_width": 12})

    code = gen.generate()

    assert "input wire [15:0] input_bus" in code
    assert "output wire [11:0] output_bus" in code


def test_verilog_generator_dense_instances():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Dense layers should instantiate sc_dense_layer_core."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 3})
    code = gen.generate()
    assert "sc_dense_layer_core" in code


def test_verilog_generator_no_layers_still_valid():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Generator should emit module wrapper even with no layers."""
    gen = VerilogGenerator()
    code = gen.generate()
    assert "module" in code
    assert "endmodule" in code
