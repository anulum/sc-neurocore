# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for research-stage async AER HDL emission

"""Verify AEREmitter and VerilogGenerator async_aer emission, Verilator lint, and simulation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from sc_neurocore.hdl_gen import AEREmitter
from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator
from tests.hdl_gen.test_verilog_generator_topology import _DENSE_STACK, live_dense_core_lint_stub


def test_async_aer_emitter_has_handshake_ports() -> None:
    """Emit standard 4-phase AER handshake ports and first_hot_index encoder."""
    emitter = AEREmitter(module_name="async_top")
    emitter.add_layer("Dense", "dense0", {"n_neurons": 4})
    code = emitter.generate()
    assert "module async_top" in code
    assert "input wire aer_ack" in code
    assert "output reg aer_req" in code
    assert "output reg [1:0] aer_addr" in code
    assert "function [1:0] first_hot_index;" in code
    assert ".N_INPUTS(8)" in code
    assert ".N_NEURONS(4)" in code
    assert ".spikes(spike_vector)" in code


def test_async_aer_emitter_derives_declared_widths() -> None:
    """Honour custom bus_width and layer output_width in AEREmitter."""
    emitter = AEREmitter(module_name="async_wide", bus_width=16)
    emitter.add_layer("Dense", "dense0", {"n_neurons": 12, "output_width": 12})

    code = emitter.generate()

    assert "input wire [15:0] input_bus" in code
    assert "output reg [3:0] aer_addr" in code
    assert "output wire [11:0] output_bus" in code
    assert "wire [11:0] spike_vector" in code
    assert "function [3:0] first_hot_index;" in code


def test_async_aer_emitter_requires_dense_neuron_count() -> None:
    """Reject Dense layers that omit n_neurons."""
    emitter = AEREmitter(module_name="async_invalid")
    emitter.add_layer("Dense", "dense0", {})
    with pytest.raises(ValueError, match="Dense layer 'dense0' requires n_neurons"):
        emitter.generate()


def test_verilog_generator_async_mode_routes_to_aer_wrapper() -> None:
    """Forward mode='async_aer' and emit_async_aer() with matching bus_width."""
    gen = VerilogGenerator(module_name="async_wrap", bus_width=12)
    gen.add_layer("Dense", "dense0", {"n_neurons": 4})
    code = gen.generate(mode="async_aer")
    assert "module async_wrap" in code
    assert "input wire [11:0] input_bus" in code
    assert "assign output_bus = spike_vector;" in code
    assert "aer_req <= 1'b1;" in code
    assert gen.emit_async_aer() == code


def test_verilog_generator_unknown_mode_raises() -> None:
    """Reject unknown generation modes fail-closed."""
    gen = VerilogGenerator()
    with pytest.raises(ValueError, match="mode must be 'sync' or 'async_aer'"):
        gen.generate(mode="invalid")


def test_async_aer_emitter_compiles_with_real_dense_core_and_verilator(tmp_path: Path) -> None:
    """Elaborate AEREmitter under Verilator -Wall and simulate with sc_dense_layer_core.v."""
    assert shutil.which("verilator") is not None
    assert shutil.which("iverilog") is not None
    assert shutil.which("vvp") is not None

    emitter = AEREmitter(module_name="async_compile", bus_width=4)
    emitter.add_layer("Dense", "dense0", {"n_neurons": 4, "stream_len": 64})
    emitter.add_layer("Dense", "dense1", {"n_neurons": 4, "stream_len": 64})
    dut_code = emitter.generate()

    stub_path = tmp_path / "sc_dense_layer_core.v"
    rtl_path = tmp_path / "async_compile.v"
    stub_path.write_text(live_dense_core_lint_stub(), encoding="utf-8")
    rtl_path.write_text(dut_code, encoding="utf-8")

    lint = subprocess.run(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "--top-module",
            "async_compile",
            str(stub_path),
            str(rtl_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert lint.returncode == 0, lint.stderr

    tb_sv = """\
`timescale 1ns/1ps
module tb_async_real;
    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg rst_n = 1'b0;
    reg [3:0] input_bus = 4'b1111;
    reg aer_ack = 1'b0;
    wire aer_req;
    wire [1:0] aer_addr;
    wire [3:0] output_bus;

    async_compile dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_bus(input_bus),
        .aer_ack(aer_ack),
        .aer_req(aer_req),
        .aer_addr(aer_addr),
        .output_bus(output_bus)
    );

    integer req_seen = 0;
    always @(negedge clk) begin
        if (aer_req) req_seen = req_seen + 1;
        aer_ack = aer_req;
    end

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (90) @(posedge clk);
        if (req_seen == 0) $fatal(1, "expected AER request from real sc_dense_layer_core");
        $display("ASYNC_REAL_DENSE_PASS reqs=%0d", req_seen);
        $finish;
    end
endmodule
"""
    tb_path = tmp_path / "tb_async_real.sv"
    sim_path = tmp_path / "async_real.vvp"
    tb_path.write_text(tb_sv, encoding="utf-8")

    comp = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-o",
            str(sim_path),
            *(str(p) for p in _DENSE_STACK),
            str(rtl_path),
            str(tb_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert comp.returncode == 0, comp.stderr
    sim = subprocess.run(
        ["vvp", "-n", str(sim_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert sim.returncode == 0, f"{sim.stdout}\n{sim.stderr}"
    assert "ASYNC_REAL_DENSE_PASS" in sim.stdout


def test_async_aer_emitter_does_not_replay_stable_spike_vector(tmp_path: Path) -> None:
    """Verify first_hot_index and replay suppression on per-index spikes."""
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    assert iverilog is not None and vvp is not None

    emitter = AEREmitter(module_name="async_replay_gate", bus_width=8)
    emitter.add_layer("Dense", "dense0", {"n_neurons": 8})
    source = (
        live_dense_core_lint_stub()
        + "\n"
        + emitter.generate()
        + """
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
    rtl_path.write_text(source, encoding="utf-8")

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run(
        [vvp, "-n", str(sim_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr


def test_async_aer_emitter_emits_zero_spike_vector_for_empty_network(tmp_path: Path) -> None:
    """Emit zero spike_vector for empty network and pass Verilator -Wall."""
    assert shutil.which("verilator") is not None
    code = AEREmitter(module_name="async_empty").generate()
    assert "assign spike_vector = 8'b0;" in code
    rtl_path = tmp_path / "async_empty.v"
    rtl_path.write_text(code, encoding="utf-8")
    lint = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "--top-module", "async_empty", str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert lint.returncode == 0, lint.stderr


def test_async_aer_emitter_rejects_non_positive_bus_width() -> None:
    """Reject zero or negative bus_width."""
    with pytest.raises(ValueError, match="bus_width must be a positive integer"):
        AEREmitter(module_name="async_bad_bus", bus_width=0)


def test_async_aer_emitter_rejects_unsupported_layer_type() -> None:
    """Reject non-Dense layers in AEREmitter."""
    emitter = AEREmitter(module_name="async_unsupported")
    emitter.add_layer("Conv", "conv0", {"n_neurons": 4})
    with pytest.raises(ValueError, match="unsupported async AER layer type 'Conv'"):
        emitter.generate()


def test_async_aer_emitter_honours_explicit_input_width() -> None:
    """Honour explicit input_width on first Dense layer."""
    emitter = AEREmitter(module_name="async_explicit_in", bus_width=8)
    emitter.add_layer("Dense", "dense0", {"n_neurons": 6, "input_width": 8})
    code = emitter.generate()
    assert "dense0_inst" in code


def test_async_aer_emitter_rejects_layer_width_mismatch() -> None:
    """Reject mismatched output_width -> next input_width and output_width != n_neurons."""
    emitter = AEREmitter(module_name="async_mismatch")
    emitter.add_layer("Dense", "dense0", {"n_neurons": 4, "output_width": 4})
    emitter.add_layer("Dense", "dense1", {"n_neurons": 6, "input_width": 7})
    with pytest.raises(ValueError, match="width mismatch"):
        emitter.generate()

    bad_out = AEREmitter(module_name="async_bad_out")
    bad_out.add_layer("Dense", "dense0", {"n_neurons": 4, "output_width": 2})
    with pytest.raises(ValueError, match="output_width"):
        bad_out.generate()
