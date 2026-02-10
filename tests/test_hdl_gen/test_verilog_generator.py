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
    assert "wire [7:0] layer_0_to_1;" in code


def test_verilog_generator_dense_instances():
    """Dense layers should instantiate sc_dense_layer_core."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 3})
    code = gen.generate()
    assert "sc_dense_layer_core" in code


def test_verilog_generator_default_neurons():
    """Missing n_neurons should default to 10."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {})
    code = gen.generate()
    assert ".NUM_NEURONS(10)" in code


def test_verilog_generator_non_dense_ignored():
    """Non-Dense layers should not emit instantiations."""
    gen = VerilogGenerator()
    gen.add_layer("Custom", "custom0", {})
    code = gen.generate()
    assert "custom0_inst" not in code


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
