
"""Module-specific tests for the precision envelope guard RTL contract."""

from pathlib import Path
import shutil
import subprocess


HDL_PATH = Path("hdl/sc_precision_envelope_guard.v")


def test_precision_envelope_guard_hdl_exposes_absolute_bound_contract() -> None:
    """The guard must check per-output absolute bounds against Q-format limits."""
    source = HDL_PATH.read_text(encoding="utf-8")

    assert "module sc_precision_envelope_guard" in source
    assert "parameter integer N_OUTPUTS = 32" in source
    assert "parameter integer OUTPUT_WIDTH = 32" in source
    assert "parameter integer BOUND_WIDTH = 48" in source
    assert "input wire [N_OUTPUTS*BOUND_WIDTH-1:0] abs_bounds_q" in source
    assert "output reg [N_OUTPUTS-1:0] violation_vector" in source
    assert "assign envelope_violation = |violation_vector" in source
    assert "bound_lane > MAX_SAFE_BOUND" in source


def test_precision_envelope_guard_hdl_flags_only_excessive_bounds(tmp_path: Path) -> None:
    """The guard must flag bounds above the signed positive Q-domain."""
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for precision envelope HDL parity")

    tb_path = tmp_path / "tb_precision_envelope_guard.v"
    sim_path = tmp_path / "precision_envelope_guard.out"
    tb_path.write_text(
        """
`timescale 1ns / 1ps

module tb_precision_envelope_guard;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg [95:0] abs_bounds_q = 96'b0;
    wire valid_out;
    wire [2:0] violation_vector;
    wire envelope_violation;

    sc_precision_envelope_guard #(
        .N_OUTPUTS(3),
        .OUTPUT_WIDTH(8),
        .BOUND_WIDTH(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .abs_bounds_q(abs_bounds_q),
        .valid_out(valid_out),
        .violation_vector(violation_vector),
        .envelope_violation(envelope_violation)
    );

    always #5 clk = ~clk;

    initial begin
        #12 rst_n = 1'b1;
        abs_bounds_q[31:0] = 32'd1;
        abs_bounds_q[63:32] = 32'd127;
        abs_bounds_q[95:64] = 32'd126;
        valid_in = 1'b1;
        #10;
        if (valid_out !== 1'b1 || violation_vector !== 3'b000 || envelope_violation !== 1'b0) begin
            $fatal(1, "safe bounds were flagged");
        end

        abs_bounds_q[31:0] = 32'd128;
        abs_bounds_q[63:32] = 32'd127;
        abs_bounds_q[95:64] = 32'd255;
        #10;
        if (violation_vector !== 3'b101 || envelope_violation !== 1'b1) begin
            $fatal(1, "excessive bounds were not flagged per lane");
        end

        valid_in = 1'b0;
        abs_bounds_q = 96'b0;
        #10;
        if (valid_out !== 1'b0 || violation_vector !== 3'b000 || envelope_violation !== 1'b0) begin
            $fatal(1, "invalid cycle must clear registered envelope result");
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
