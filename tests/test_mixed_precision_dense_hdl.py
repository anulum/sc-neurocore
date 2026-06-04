# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for mixed-precision dense HDL reference

"""Module-specific tests for the mixed-precision dense HDL contract."""

from pathlib import Path
import shutil
import subprocess


HDL_PATH = Path("hdl/sc_mixed_precision_dense.v")


def test_mixed_precision_dense_hdl_exposes_q88_q1616_contract() -> None:
    """The reference RTL must keep compact weights and widened accumulators explicit."""
    source = HDL_PATH.read_text(encoding="utf-8")

    assert "module sc_mixed_precision_dense" in source
    assert "parameter integer WEIGHT_WIDTH = 16" in source
    assert "parameter integer INPUT_WIDTH = 32" in source
    assert "parameter integer ACCUM_WIDTH = 32" in source
    assert "parameter integer WEIGHT_FRAC = 8" in source
    assert "weights_q88" in source
    assert "inputs_q1616" in source
    assert "outputs_q1616" in source
    assert "output reg [N_OUTPUTS-1:0] overflow_vector" in source


def test_mixed_precision_dense_hdl_saturates_instead_of_silent_wraparound() -> None:
    """Overflow must become a telemetry bit and saturated code, not corrupted output."""
    source = HDL_PATH.read_text(encoding="utf-8")

    assert "overflow_next = 1'b1" in source
    assert "overflow_vector_next[output_idx] = 1'b1" in source
    assert "outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MAX" in source
    assert "outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MIN" in source
    assert "scaled_sum = sum >>> WEIGHT_FRAC" in source


def test_mixed_precision_dense_hdl_reports_overflow_per_output_lane(tmp_path: Path) -> None:
    """Lane telemetry must identify positive and negative saturation independently."""
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for mixed dense HDL parity")

    tb_path = tmp_path / "tb_mixed_precision_dense.v"
    sim_path = tmp_path / "mixed_precision_dense.out"
    tb_path.write_text(
        """
`timescale 1ns / 1ps

module tb_mixed_precision_dense;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg signed [95:0] weights_q88 = 96'b0;
    reg signed [63:0] inputs_q1616 = 64'b0;
    wire valid_out;
    wire signed [95:0] outputs_q1616;
    wire [2:0] overflow_vector;
    wire overflow;

    sc_mixed_precision_dense #(
        .N_INPUTS(2),
        .N_OUTPUTS(3),
        .WEIGHT_WIDTH(16),
        .INPUT_WIDTH(32),
        .ACCUM_WIDTH(32),
        .WEIGHT_FRAC(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .weights_q88(weights_q88),
        .inputs_q1616(inputs_q1616),
        .valid_out(valid_out),
        .outputs_q1616(outputs_q1616),
        .overflow_vector(overflow_vector),
        .overflow(overflow)
    );

    always #5 clk = ~clk;

    initial begin
        #12 rst_n = 1'b1;
        weights_q88[15:0] = 16'sd256;
        weights_q88[31:16] = -16'sd256;
        weights_q88[47:32] = 16'sd32767;
        weights_q88[63:48] = 16'sd32767;
        weights_q88[79:64] = 16'sh8000;
        weights_q88[95:80] = 16'sh8000;
        inputs_q1616[31:0] = 32'sd2147418112;
        inputs_q1616[63:32] = 32'sd2147418112;
        valid_in = 1'b1;
        #10;

        if (valid_out !== 1'b1) begin
            $fatal(1, "valid output not registered");
        end
        if (outputs_q1616[31:0] !== 32'sd0) begin
            $fatal(1, "cancelling safe lane changed");
        end
        if (outputs_q1616[63:32] !== 32'sh7fffffff) begin
            $fatal(1, "positive saturated lane did not clamp to max");
        end
        if (outputs_q1616[95:64] !== 32'sh80000000) begin
            $fatal(1, "negative saturated lane did not clamp to min");
        end
        if (overflow !== 1'b1 || overflow_vector !== 3'b110) begin
            $fatal(1, "lane overflow vector mismatch");
        end

        valid_in = 1'b0;
        #10;
        if (valid_out !== 1'b0 || overflow !== 1'b0 || overflow_vector !== 3'b000) begin
            $fatal(1, "invalid cycle did not clear overflow telemetry");
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


def test_mixed_precision_dense_reports_per_output_abs_bounds(tmp_path: Path) -> None:
    import shutil
    import subprocess
    import textwrap

    import pytest

    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog/vvp unavailable")

    testbench = tmp_path / "mixed_precision_dense_bounds_tb.v"
    executable = tmp_path / "mixed_precision_dense_bounds_tb.out"
    testbench.write_text(
        textwrap.dedent(
            r"""
            `timescale 1ns / 1ps

            module mixed_precision_dense_bounds_tb;
                localparam integer N_INPUTS = 2;
                localparam integer N_OUTPUTS = 3;
                localparam integer WEIGHT_WIDTH = 16;
                localparam integer INPUT_WIDTH = 32;
                localparam integer ACCUM_WIDTH = 32;
                localparam integer BOUND_WIDTH = 64;

                reg clk = 1'b0;
                reg rst_n = 1'b0;
                reg valid_in = 1'b0;
                reg signed [N_OUTPUTS*N_INPUTS*WEIGHT_WIDTH-1:0] weights_q88 = 0;
                reg signed [N_INPUTS*INPUT_WIDTH-1:0] inputs_q1616 = 0;
                wire valid_out;
                wire signed [N_OUTPUTS*ACCUM_WIDTH-1:0] outputs_q1616;
                wire [N_OUTPUTS*BOUND_WIDTH-1:0] abs_bounds_q1616;
                wire [N_OUTPUTS-1:0] overflow_vector;
                wire overflow;

                sc_mixed_precision_dense #(
                    .N_INPUTS(N_INPUTS),
                    .N_OUTPUTS(N_OUTPUTS),
                    .WEIGHT_WIDTH(WEIGHT_WIDTH),
                    .INPUT_WIDTH(INPUT_WIDTH),
                    .ACCUM_WIDTH(ACCUM_WIDTH),
                    .WEIGHT_FRAC(8),
                    .BOUND_WIDTH(BOUND_WIDTH)
                ) dut (
                    .clk(clk),
                    .rst_n(rst_n),
                    .valid_in(valid_in),
                    .weights_q88(weights_q88),
                    .inputs_q1616(inputs_q1616),
                    .valid_out(valid_out),
                    .outputs_q1616(outputs_q1616),
                    .abs_bounds_q1616(abs_bounds_q1616),
                    .overflow_vector(overflow_vector),
                    .overflow(overflow)
                );

                always #5 clk = ~clk;

                initial begin
                    inputs_q1616[0*INPUT_WIDTH +: INPUT_WIDTH] = 32'sd2147418112;
                    inputs_q1616[1*INPUT_WIDTH +: INPUT_WIDTH] = 32'sd2147418112;
                    weights_q88[0*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 16'sd256;
                    weights_q88[1*WEIGHT_WIDTH +: WEIGHT_WIDTH] = -16'sd256;
                    weights_q88[2*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 16'sd32767;
                    weights_q88[3*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 16'sd32767;
                    weights_q88[4*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 16'sh8000;
                    weights_q88[5*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 16'sh8000;

                    #12 rst_n = 1'b1;
                    #8 valid_in = 1'b1;
                    #10 valid_in = 1'b0;
                    #10;

                    if (valid_out !== 1'b1) begin
                        $display("expected valid_out");
                        $finish(1);
                    end
                    if (overflow_vector !== 3'b110 || overflow !== 1'b1) begin
                        $display("unexpected overflow telemetry vector=%b overflow=%b", overflow_vector, overflow);
                        $finish(1);
                    end
                    if (abs_bounds_q1616[0*BOUND_WIDTH +: BOUND_WIDTH] !== 64'd4294836224) begin
                        $display("unexpected cancellation bound %0d", abs_bounds_q1616[0*BOUND_WIDTH +: BOUND_WIDTH]);
                        $finish(1);
                    end
                    if (abs_bounds_q1616[1*BOUND_WIDTH +: BOUND_WIDTH] !== 64'd549722259968) begin
                        $display("unexpected positive saturation bound %0d", abs_bounds_q1616[1*BOUND_WIDTH +: BOUND_WIDTH]);
                        $finish(1);
                    end
                    if (abs_bounds_q1616[2*BOUND_WIDTH +: BOUND_WIDTH] !== 64'd549739036672) begin
                        $display("unexpected negative saturation bound %0d", abs_bounds_q1616[2*BOUND_WIDTH +: BOUND_WIDTH]);
                        $finish(1);
                    end
                    $finish(0);
                end
            endmodule
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-o",
            str(executable),
            str(testbench),
            "hdl/sc_mixed_precision_dense.v",
        ],
        check=True,
    )
    subprocess.run(["vvp", str(executable)], check=True)
