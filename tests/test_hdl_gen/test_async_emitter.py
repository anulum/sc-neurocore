# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for research-stage async AER HDL emission

from __future__ import annotations

import shutil
import subprocess

from sc_neurocore.hdl_gen import AEREmitter
from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator


def _stub_dense_layer() -> str:
    return """module sc_dense_layer_core #(parameter NUM_NEURONS = 8) (
    input wire clk,
    input wire rst_n,
    input wire [7:0] input_bus,
    output wire [7:0] output_bus
);
assign output_bus = input_bus;
endmodule
"""


def test_async_aer_emitter_has_handshake_ports() -> None:
    emitter = AEREmitter(module_name="async_top")
    emitter.add_layer("Dense", "dense0", {"n_neurons": 4})
    code = emitter.generate()
    assert "module async_top" in code
    assert "input wire aer_ack" in code
    assert "output reg aer_req" in code
    assert "output reg [7:0] aer_addr" in code
    assert "function [7:0] first_hot_index;" in code


def test_verilog_generator_async_mode_routes_to_aer_wrapper() -> None:
    gen = VerilogGenerator(module_name="async_wrap")
    gen.add_layer("Dense", "dense0", {"n_neurons": 4})
    code = gen.generate(mode="async_aer")
    assert "module async_wrap" in code
    assert "assign output_bus = spike_vector;" in code
    assert "aer_req <= 1'b1;" in code


def test_verilog_generator_unknown_mode_raises() -> None:
    gen = VerilogGenerator()
    try:
        gen.generate(mode="invalid")
    except ValueError as exc:
        assert "mode must be 'sync' or 'async_aer'" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for invalid mode")


def test_async_aer_emitter_smoke_compiles_with_iverilog(tmp_path) -> None:
    iverilog = shutil.which("iverilog")
    if iverilog is None:
        raise AssertionError("iverilog must be available for HDL smoke tests")

    emitter = AEREmitter(module_name="async_compile")
    emitter.add_layer("Dense", "dense0", {"n_neurons": 4})
    emitter.add_layer("Dense", "dense1", {"n_neurons": 4})
    source = _stub_dense_layer() + "\n" + emitter.generate()

    rtl_path = tmp_path / "async_compile.v"
    rtl_path.write_text(source)

    result = subprocess.run(
        [iverilog, "-g2012", "-t", "null", str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
