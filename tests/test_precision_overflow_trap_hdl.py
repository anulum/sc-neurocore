# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for precision overflow trap HDL

"""Module-specific tests for the precision overflow trap RTL contract."""

from pathlib import Path
import shutil
import subprocess


HDL_PATH = Path("hdl/sc_precision_overflow_trap.v")


def test_precision_overflow_trap_hdl_exposes_sticky_vector_contract() -> None:
    """The trap module must expose a latchable vector and explicit host clear."""
    source = HDL_PATH.read_text(encoding="utf-8")

    assert "module sc_precision_overflow_trap" in source
    assert "parameter integer TRAP_WIDTH = 1" in source
    assert "input wire clear_trap" in source
    assert "input wire [TRAP_WIDTH-1:0] overflow_in" in source
    assert "output reg [TRAP_WIDTH-1:0] trap_vector" in source
    assert "output wire [TRAP_WIDTH-1:0] trap_event_vector" in source
    assert "output wire trap_event" in source
    assert "output wire trap_latched" in source
    assert "trap_accepting = rst_n & ~clear_trap" in source
    assert "trap_event_vector = trap_accepting ? overflow_in" in source
    assert "trap_vector <= trap_vector | trap_event_vector" in source
    assert "a_no_silent_precision_overflow" in source
    assert "a_sticky_precision_overflow" in source


def test_precision_overflow_trap_hdl_latches_clears_and_accumulates(tmp_path: Path) -> None:
    """A transient overflow pulse must persist until the host clears the trap."""
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for precision trap HDL parity")

    tb_path = tmp_path / "tb_precision_overflow_trap.v"
    sim_path = tmp_path / "precision_overflow_trap.out"
    tb_path.write_text(
        """
`timescale 1ns / 1ps

module tb_precision_overflow_trap;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg clear_trap = 1'b0;
    reg [2:0] overflow_in = 3'b000;
    wire [2:0] trap_vector;
    wire [2:0] trap_event_vector;
    wire trap_event;
    wire trap_latched;

    sc_precision_overflow_trap #(
        .TRAP_WIDTH(3)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear_trap(clear_trap),
        .overflow_in(overflow_in),
        .trap_vector(trap_vector),
        .trap_event_vector(trap_event_vector),
        .trap_event(trap_event),
        .trap_latched(trap_latched)
    );

    always #5 clk = ~clk;

    initial begin
        #12 rst_n = 1'b1;
        #10 overflow_in = 3'b010;
        #1;
        if (trap_event_vector !== 3'b010 || trap_event !== 1'b1) begin
            $fatal(1, "overflow event was not exposed in the same cycle");
        end
        #9;
        if (trap_vector !== 3'b010 || trap_latched !== 1'b1) begin
            $fatal(1, "overflow pulse was not retained after latch edge");
        end
        #10 overflow_in = 3'b000;
        #1;
        if (trap_event_vector !== 3'b000 || trap_event !== 1'b0) begin
            $fatal(1, "idle cycle leaked a trap event");
        end
        #10;
        if (trap_vector !== 3'b010 || trap_latched !== 1'b1) begin
            $fatal(1, "overflow pulse was not retained");
        end

        clear_trap = 1'b1;
        overflow_in = 3'b111;
        #1;
        if (trap_event_vector !== 3'b000 || trap_event !== 1'b0) begin
            $fatal(1, "clear must suppress same-cycle overflow event");
        end
        #9;
        if (trap_vector !== 3'b000 || trap_latched !== 1'b0) begin
            $fatal(1, "clear must dominate concurrent overflow");
        end

        clear_trap = 1'b0;
        overflow_in = 3'b100;
        #1;
        if (trap_event_vector !== 3'b100 || trap_event !== 1'b1) begin
            $fatal(1, "first accumulated overflow event was not exposed");
        end
        #9;
        overflow_in = 3'b001;
        #1;
        if (trap_event_vector !== 3'b001 || trap_event !== 1'b1) begin
            $fatal(1, "second accumulated overflow event was not exposed");
        end
        #9;
        overflow_in = 3'b000;
        #10;
        if (trap_vector !== 3'b101 || trap_latched !== 1'b1) begin
            $fatal(1, "trap vector did not accumulate overflow lanes");
        end

        $display("PASS");
        $finish;
    end
endmodule
""",
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(HDL_PATH), str(tb_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run([vvp, str(sim_path)], check=False, capture_output=True, text=True)
    assert run_result.returncode == 0, run_result.stderr
    assert "PASS" in run_result.stdout
