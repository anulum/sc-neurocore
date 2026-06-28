# SPDX-License-Identifier: AGPL-3.0-or-later
"""UltraScale+ dense-folding resource and HDL contract."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ultrascale_dense_folding import plan_dense_fold


def test_dense_fold_plan_fits_zu3eg_shd_scale_contract() -> None:
    plan = plan_dense_fold(n_inputs=64, n_outputs=32, dsp_budget=360)
    assert plan.mac_count == 2048
    assert plan.output_parallelism == 5
    assert plan.input_parallelism == 64
    assert plan.dsp_per_cycle == 320
    assert plan.output_fold_factor == 7
    assert plan.input_fold_factor == 1
    assert plan.compute_cycles == 7
    assert plan.fold_required is True
    assert plan.fits_dsp_budget is True


def test_dense_fold_plan_rejects_negative_dimensions() -> None:
    try:
        plan_dense_fold(n_inputs=-1, n_outputs=32, dsp_budget=360)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative dense dimensions must be rejected")


def test_rust_dense_fold_contracts_pass(cargo_lib_test) -> None:
    completed = cargo_lib_test("dense_fold")
    assert completed.returncode == 0


def test_dense_folded_q88_core_simulates_parallel_output_groups(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for dense-folded HDL parity")

    tb = tmp_path / "tb_dense_folded_q88_core.v"
    tb.write_text(
        r"""
`timescale 1ns / 1ps

module tb_dense_folded_q88_core;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start_pulse = 1'b0;
    reg [63:0] x_input_fp;
    reg [191:0] weight_fp;
    wire [2:0] spikes;
    wire step_valid;
    wire run_done;
    wire running;
    wire overflow;
    wire [31:0] compute_cycle_count;

    always #5 clk = ~clk;

    sc_dense_folded_q88_core #(
        .N_INPUTS(4),
        .N_NEURONS(3),
        .DATA_WIDTH(16),
        .FRAC_BITS(8),
        .PARALLEL_NEURONS(2),
        .THRESHOLD_Q(48'sd256)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_pulse(start_pulse),
        .x_input_fp(x_input_fp),
        .weight_fp(weight_fp),
        .cfg_leak(16'd0),
        .cfg_gain(16'd256),
        .spikes(spikes),
        .step_valid(step_valid),
        .run_done(run_done),
        .running(running),
        .overflow(overflow),
        .compute_cycle_count(compute_cycle_count)
    );

    initial begin
        x_input_fp = {16'sd0, 16'sd0, 16'sd0, 16'sd256};
        weight_fp = {
            16'sd0, 16'sd0, 16'sd0, 16'sd512,
            16'sd0, 16'sd0, 16'sd0, 16'sd0,
            16'sd0, 16'sd0, 16'sd0, 16'sd256
        };
        #12 rst_n = 1'b1;
        #10 start_pulse = 1'b1;
        #10 start_pulse = 1'b0;
        wait (run_done == 1'b1);
        #2;
        if (spikes !== 3'b101) begin
            $display("unexpected spikes=%b", spikes);
            $finish(1);
        end
        if (compute_cycle_count !== 32'd2) begin
            $display("unexpected compute cycles=%0d", compute_cycle_count);
            $finish(1);
        end
        if (overflow !== 1'b0) begin
            $display("unexpected overflow");
            $finish(1);
        end
        $finish(0);
    end
endmodule
""",
        encoding="utf-8",
    )
    sim_path = tmp_path / "dense_folded.out"
    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), "hdl/sc_dense_folded_q88_core.v", str(tb)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run([vvp, str(sim_path)], check=False, capture_output=True, text=True)
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
