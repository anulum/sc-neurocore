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
    assert "output reg [1:0] aer_addr" in code
    assert "function [1:0] first_hot_index;" in code


def test_async_aer_emitter_derives_declared_widths() -> None:
    emitter = AEREmitter(module_name="async_wide", bus_width=16)
    emitter.add_layer("Dense", "dense0", {"n_neurons": 12, "output_width": 12})

    code = emitter.generate()

    assert "input wire [15:0] input_bus" in code
    assert "output reg [3:0] aer_addr" in code
    assert "output wire [11:0] output_bus" in code
    assert "wire [11:0] spike_vector" in code
    assert "function [3:0] first_hot_index;" in code


def test_async_aer_emitter_requires_dense_neuron_count() -> None:
    emitter = AEREmitter(module_name="async_invalid")
    emitter.add_layer("Dense", "dense0", {})

    try:
        emitter.generate()
    except ValueError as exc:
        assert "Dense layer 'dense0' requires n_neurons" in str(exc)
    else:
        raise AssertionError("Expected ValueError for omitted n_neurons")


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


def test_async_aer_emitter_does_not_replay_stable_spike_vector(tmp_path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for HDL simulation tests")

    emitter = AEREmitter(module_name="async_replay_gate", bus_width=8)
    emitter.add_layer("Dense", "dense0", {"n_neurons": 8})
    source = (
        _stub_dense_layer()
        + "\n"
        + emitter.generate()
        + r"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [7:0] input_bus = 8'b0;
    reg aer_ack = 1'b0;
    wire aer_req;
    wire [2:0] aer_addr;
    wire [7:0] output_bus;

    async_replay_gate dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_bus(input_bus),
        .aer_ack(aer_ack),
        .aer_req(aer_req),
        .aer_addr(aer_addr),
        .output_bus(output_bus)
    );

    integer event_count = 0;
    integer first_addr = -1;
    integer second_addr = -1;
    integer third_addr = -1;
    reg prev_req = 1'b0;

    always #5 clk = ~clk;

    always @(negedge clk) begin
        if (aer_req && !prev_req) begin
            event_count = event_count + 1;
            if (event_count == 1) first_addr = aer_addr;
            if (event_count == 2) second_addr = aer_addr;
            if (event_count == 3) third_addr = aer_addr;
        end
        prev_req = aer_req;
        aer_ack = aer_req;
    end

    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        input_bus = 8'b00001000;
        repeat (8) @(posedge clk);

        input_bus = 8'b00010000;
        repeat (8) @(posedge clk);

        input_bus = 8'b00000000;
        repeat (4) @(posedge clk);

        input_bus = 8'b00001000;
        repeat (8) @(posedge clk);

        if (event_count != 3) begin
            $fatal(1, "expected exactly 3 unique AER events, observed %0d", event_count);
        end
        if (first_addr != 3 || second_addr != 4 || third_addr != 3) begin
            $fatal(
                1,
                "unexpected AER addresses: first=%0d second=%0d third=%0d",
                first_addr,
                second_addr,
                third_addr
            );
        end
        $finish(0);
    end
endmodule
"""
    )

    rtl_path = tmp_path / "async_replay_gate.v"
    sim_path = tmp_path / "async_replay_gate.out"
    rtl_path.write_text(source)

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run(
        [vvp, str(sim_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr


def test_async_aer_emitter_emits_zero_spike_vector_for_empty_network() -> None:
    code = AEREmitter(module_name="async_empty").generate()
    assert "assign spike_vector = 8'b0;" in code


def test_async_aer_emitter_rejects_non_positive_bus_width() -> None:
    try:
        AEREmitter(module_name="async_bad_bus", bus_width=0)
    except ValueError as exc:
        assert "bus_width must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive bus_width")


def test_async_aer_emitter_rejects_unsupported_layer_type() -> None:
    emitter = AEREmitter(module_name="async_unsupported")
    emitter.add_layer("Conv", "conv0", {"n_neurons": 4})
    try:
        emitter.generate()
    except ValueError as exc:
        assert "unsupported async AER layer type 'Conv'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported layer type")


def test_async_aer_emitter_honours_explicit_input_width() -> None:
    emitter = AEREmitter(module_name="async_explicit_in", bus_width=8)
    emitter.add_layer("Dense", "dense0", {"n_neurons": 6, "input_width": 8})
    code = emitter.generate()
    assert "dense0_inst" in code


def test_async_aer_emitter_rejects_layer_width_mismatch() -> None:
    emitter = AEREmitter(module_name="async_mismatch")
    emitter.add_layer("Dense", "dense0", {"n_neurons": 4, "output_width": 4})
    emitter.add_layer("Dense", "dense1", {"n_neurons": 6, "input_width": 7})
    try:
        emitter.generate()
    except ValueError as exc:
        assert "width mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for width mismatch")
