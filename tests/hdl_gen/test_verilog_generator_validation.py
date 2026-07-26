# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog generator validation tests

"""Width, layer topology, neuron count, and source-seed validation."""

import pytest

from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator


def test_verilog_generator_rejects_mismatched_dense_widths():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Adjacent Dense layers must agree on the inter-layer bus width."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 5, "output_width": 5})
    gen.add_layer("Dense", "dense1", {"n_neurons": 3, "input_width": 7})

    with pytest.raises(ValueError, match="dense0 -> dense1 width mismatch"):
        gen.generate()


def test_verilog_generator_requires_dense_neuron_count():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Dense sync generation must not invent omitted neuron counts."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {})
    with pytest.raises(ValueError, match="Dense layer 'dense0' requires n_neurons"):
        gen.generate()


def test_verilog_generator_rejects_unsupported_sync_layer():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Unsupported sync layers must fail closed instead of being silently dropped."""
    gen = VerilogGenerator()
    gen.add_layer("Custom", "custom0", {})
    with pytest.raises(ValueError, match="unsupported sync layer type 'Custom'"):
        gen.generate()


def test_source_seed_rejects_non_integer_seed():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """A stochastic-source seed must be a real integer, not a bool or string."""
    from sc_neurocore.hdl_gen.verilog_generator import _source_seed

    with pytest.raises(ValueError, match="seed must be an integer"):
        _source_seed({"seed": True}, default=0)
    with pytest.raises(ValueError, match="seed must be an integer"):
        _source_seed({"seed": "not-an-int"}, default=0)
