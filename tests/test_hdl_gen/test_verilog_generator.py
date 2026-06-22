# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for VerilogGenerator code emission and file output

"""Tests for VerilogGenerator code emission and file output."""

import os
import time

import pytest

from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_verilog_generator_module_name():
    """Generated code should contain module name."""
    gen = VerilogGenerator(module_name="my_top")
    code = gen.generate()
    assert "module my_top" in code


def test_verilog_generator_add_layer():
    """add_layer should register layers in order."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "layer0", {"n_neurons": 4})
    assert len(gen.layers) == 1
    assert gen.layers[0]["name"] == "layer0"


def test_verilog_generator_single_layer_wiring():
    """Single Dense layer should connect input_bus to output_bus."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 4})
    code = gen.generate()
    assert ".input_bus(input_bus)" in code
    assert ".output_bus(output_bus)" in code


def test_verilog_generator_two_layers_wires():
    """Two Dense layers should include inter-layer wire."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 4})
    gen.add_layer("Dense", "dense1", {"n_neurons": 4})
    code = gen.generate()
    assert "wire [3:0] layer_0_to_1;" in code


def test_verilog_generator_derives_declared_bus_widths():
    """Sync generator should not force every top-level bus to eight bits."""
    gen = VerilogGenerator(module_name="wide_top", bus_width=16)
    gen.add_layer("Dense", "dense0", {"n_neurons": 12, "output_width": 12})

    code = gen.generate()

    assert "input wire [15:0] input_bus" in code
    assert "output wire [11:0] output_bus" in code


def test_verilog_generator_rejects_mismatched_dense_widths():
    """Adjacent Dense layers must agree on the inter-layer bus width."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 5, "output_width": 5})
    gen.add_layer("Dense", "dense1", {"n_neurons": 3, "input_width": 7})

    with pytest.raises(ValueError, match="dense0 -> dense1 width mismatch"):
        gen.generate()


def test_verilog_generator_dense_instances():
    """Dense layers should instantiate sc_dense_layer_core."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 3})
    code = gen.generate()
    assert "sc_dense_layer_core" in code


def test_verilog_generator_requires_dense_neuron_count():
    """Dense sync generation must not invent omitted neuron counts."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {})
    with pytest.raises(ValueError, match="Dense layer 'dense0' requires n_neurons"):
        gen.generate()


def test_verilog_generator_rejects_unsupported_sync_layer():
    """Unsupported sync layers must fail closed instead of being silently dropped."""
    gen = VerilogGenerator()
    gen.add_layer("Custom", "custom0", {})
    with pytest.raises(ValueError, match="unsupported sync layer type 'Custom'"):
        gen.generate()


def test_verilog_generator_generate_routes_stochastic_source_layers():
    """Stochastic source layers should emit their standalone source modules."""
    gen = VerilogGenerator()
    gen.add_layer("StochasticSource", "rng_lfsr", {"source_type": "LFSR", "seed": 0xBEEF})
    gen.add_layer("StochasticSource", "rng_sobol", {"source_type": "Sobol", "seed": 0x0042})

    code = gen.generate()

    assert "module rng_lfsr" in code
    assert "16'hBEEF" in code
    assert "module rng_sobol" in code
    assert "16'h0042" in code


def test_verilog_generator_no_layers_still_valid():
    """Generator should emit module wrapper even with no layers."""
    gen = VerilogGenerator()
    code = gen.generate()
    assert "module" in code
    assert "endmodule" in code


def test_verilog_generator_save_to_file(tmp_path):
    """save_to_file should write the generated Verilog."""
    gen = VerilogGenerator(module_name="save_top")
    gen.add_layer("Dense", "dense0", {"n_neurons": 2})
    path = tmp_path / "top.v"
    gen.save_to_file(str(path))
    assert path.exists()
    contents = path.read_text()
    assert "module save_top" in contents


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_verilog_generator_perf_small():
    """Benchmark generating code for a small network."""
    gen = VerilogGenerator()
    for i in range(5):
        gen.add_layer("Dense", f"dense{i}", {"n_neurons": 8})
    start = time.perf_counter()
    _ = gen.generate()
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


def test_save_to_file_reraises_oserror(tmp_path):
    """save_to_file should log and re-raise when the target path is unwritable."""
    gen = VerilogGenerator(module_name="io_fail")
    # The parent directory does not exist, so open() raises an OSError.
    bad_path = tmp_path / "missing_subdir" / "out.v"
    with pytest.raises(OSError):
        gen.save_to_file(str(bad_path))


def test_source_seed_rejects_non_integer_seed():
    """A stochastic-source seed must be a real integer, not a bool or string."""
    from sc_neurocore.hdl_gen.verilog_generator import _source_seed

    with pytest.raises(ValueError, match="seed must be an integer"):
        _source_seed({"seed": True}, default=0)
    with pytest.raises(ValueError, match="seed must be an integer"):
        _source_seed({"seed": "not-an-int"}, default=0)
