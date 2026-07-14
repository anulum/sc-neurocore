# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AXI4-Lite live-control RTL execution contracts

"""Compile and execute the generated AXI4-Lite live-control core."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from sc_neurocore.compiler.live_control import MMIOUpdateSpec, ParameterBankSpec, TrapSpec
from sc_neurocore.hdl_gen.bus_interface import generate_live_parameter_bank


def test_live_parameter_bank_rtl_compiles_with_iverilog(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    if iverilog is None:
        raise AssertionError("iverilog must be available for live-control RTL compile parity")

    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        control_base_address_bytes=0x100,
        banks=(
            ParameterBankSpec(
                bank_name="weights",
                start_address_bytes=0x2000,
                parameter_count=2,
                parameter_names=("w0", "w1"),
                q_format="Q8.8",
            ),
        ),
        trap=TrapSpec(max_flags=4),
    )
    source_path = tmp_path / "sc_live_params.sv"
    sim_path = tmp_path / "sc_live_params.out"
    source_path.write_text(
        generate_live_parameter_bank(spec, module_name="sc_live_params"),
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert compile_result.returncode == 0, compile_result.stderr


def test_live_parameter_bank_reads_back_committed_active_word(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError(
            "iverilog and vvp must be available for live-control readback simulation"
        )

    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        control_base_address_bytes=0x100,
        banks=(
            ParameterBankSpec(
                bank_name="weights",
                start_address_bytes=0x2000,
                parameter_count=1,
                parameter_names=("w0",),
                q_format="Q8.8",
            ),
        ),
        trap=TrapSpec(max_flags=2),
    )
    source_path = tmp_path / "sc_live_params.sv"
    tb_path = tmp_path / "tb_sc_live_params.sv"
    sim_path = tmp_path / "sc_live_params.out"
    source_path.write_text(
        generate_live_parameter_bank(spec, module_name="sc_live_params"),
        encoding="utf-8",
    )
    tb_path.write_text(
        """
module tb_sc_live_params;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [31:0] awaddr = 32'd0;
    reg awvalid = 1'b0;
    wire awready;
    reg [31:0] wdata = 32'd0;
    reg [3:0] wstrb = 4'hF;
    reg wvalid = 1'b0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    reg bready = 1'b0;
    reg [31:0] araddr = 32'd0;
    reg arvalid = 1'b0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    reg rready = 1'b0;
    reg [5:0] external_trap_vector = 6'b000000;
    wire trap_latched;
    wire [5:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
    wire checksum_mismatch_pulse;
    wire invalid_selection_pulse;
    wire read_only_bank_pulse;
    wire partial_write_pulse;
    wire shadow_loaded;
    wire [15:0] parameter_words;

    always #5 clk = ~clk;

    sc_live_params dut (
        .S_AXI_ACLK(clk),
        .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(awaddr),
        .S_AXI_AWVALID(awvalid),
        .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata),
        .S_AXI_WSTRB(wstrb),
        .S_AXI_WVALID(wvalid),
        .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp),
        .S_AXI_BVALID(bvalid),
        .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr),
        .S_AXI_ARVALID(arvalid),
        .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata),
        .S_AXI_RRESP(rresp),
        .S_AXI_RVALID(rvalid),
        .S_AXI_RREADY(rready),
        .trap_vector(external_trap_vector),
        .trap_latched(trap_latched),
        .trap_status_vector(trap_status_vector),
        .staged_overflow(staged_overflow),
        .staged_underflow(staged_underflow),
        .update_pulse(update_pulse),
        .apply_pulse(apply_pulse),
        .rollback_pulse(rollback_pulse),
        .trap_clear_pulse(trap_clear_pulse),
        .checksum_mismatch_pulse(checksum_mismatch_pulse),
        .invalid_selection_pulse(invalid_selection_pulse),
        .read_only_bank_pulse(read_only_bank_pulse),
        .partial_write_pulse(partial_write_pulse),
        .shadow_loaded(shadow_loaded),
        .parameter_words(parameter_words)
    );

    task axi_write;
        input [31:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            awaddr = addr;
            wdata = data;
            awvalid = 1'b1;
            wvalid = 1'b1;
            bready = 1'b1;
            @(negedge clk);
            awvalid = 1'b0;
            wvalid = 1'b0;
            @(negedge clk);
            if (bresp !== 2'b00) begin
                $display("unexpected AXI write response at %h", addr);
                $finish(1);
            end
            bready = 1'b0;
        end
    endtask

    task axi_read_expect;
        input [31:0] addr;
        input [31:0] expected;
        begin
            @(negedge clk);
            araddr = addr;
            arvalid = 1'b1;
            rready = 1'b1;
            @(negedge clk);
            arvalid = 1'b0;
            @(negedge clk);
            if (rresp !== 2'b00 || rdata !== expected) begin
                $display("AXI readback mismatch at %h expected=%h actual=%h response=%b", addr, expected, rdata, rresp);
                $finish(2);
            end
            rready = 1'b0;
        end
    endtask

    task axi_read_expect_error;
        input [31:0] addr;
        begin
            @(negedge clk);
            araddr = addr;
            arvalid = 1'b1;
            rready = 1'b1;
            @(negedge clk);
            arvalid = 1'b0;
            @(negedge clk);
            if (rresp !== 2'b10) begin
                $display("AXI invalid readback did not fail closed at %h response=%b data=%h", addr, rresp, rdata);
                $finish(6);
            end
            rready = 1'b0;
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        axi_write(32'h108, 32'd0);
        axi_write(32'h10C, 32'd0);
        axi_write(32'h110, 32'h00001234);
        axi_write(32'h114, 32'h00000000);
        axi_write(32'h120, 32'h1D7D9B35);
        axi_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b1 || parameter_words !== 16'h0000) begin
            $display("shadow load failed or active bank mutated before commit");
            $finish(3);
        end

        axi_write(32'h100, 32'h00000002);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words !== 16'h1234) begin
            $display("commit failed before readback parameter=%h shadow=%b", parameter_words, shadow_loaded);
            $finish(4);
        end
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000 || staged_overflow !== 1'b0 || staged_underflow !== 1'b0) begin
            $display("valid commit raised trap status=%b", trap_status_vector);
            $finish(5);
        end

        axi_read_expect(32'h124, 32'h00001234);
        axi_read_expect(32'h128, 32'h00000000);

        axi_write(32'h10C, 32'd3);
        axi_read_expect_error(32'h124);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b1 || trap_status_vector[3] !== 1'b1 || invalid_selection_pulse !== 1'b1) begin
            $display("invalid active-readback selection did not raise sticky trap status=%b pulse=%b", trap_status_vector, invalid_selection_pulse);
            $finish(7);
        end

        $finish(0);
    end
endmodule
""",
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(source_path), str(tb_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [vvp, str(sim_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr


def test_live_parameter_bank_latches_generated_range_traps(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for live-control trap simulation")

    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        control_base_address_bytes=0x100,
        banks=(
            ParameterBankSpec(
                bank_name="weights",
                start_address_bytes=0x2000,
                parameter_count=1,
                parameter_names=("w0",),
                q_format="Q8.8",
            ),
        ),
        trap=TrapSpec(max_flags=2),
    )
    source_path = tmp_path / "sc_live_params.sv"
    tb_path = tmp_path / "tb_sc_live_params.sv"
    sim_path = tmp_path / "sc_live_params.out"
    source_path.write_text(
        generate_live_parameter_bank(spec, module_name="sc_live_params"),
        encoding="utf-8",
    )
    tb_path.write_text(
        """
module tb_sc_live_params;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [31:0] awaddr = 32'd0;
    reg awvalid = 1'b0;
    wire awready;
    reg [31:0] wdata = 32'd0;
    reg [3:0] wstrb = 4'hF;
    reg wvalid = 1'b0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    reg bready = 1'b0;
    reg [31:0] araddr = 32'd0;
    reg arvalid = 1'b0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    reg rready = 1'b0;
    reg [5:0] external_trap_vector = 6'b000000;
    wire trap_latched;
    wire [5:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
    wire checksum_mismatch_pulse;
    wire invalid_selection_pulse;
    wire read_only_bank_pulse;
    wire partial_write_pulse;
    wire shadow_loaded;
    wire [15:0] parameter_words;

    always #5 clk = ~clk;

    sc_live_params dut (
        .S_AXI_ACLK(clk),
        .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(awaddr),
        .S_AXI_AWVALID(awvalid),
        .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata),
        .S_AXI_WSTRB(wstrb),
        .S_AXI_WVALID(wvalid),
        .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp),
        .S_AXI_BVALID(bvalid),
        .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr),
        .S_AXI_ARVALID(arvalid),
        .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata),
        .S_AXI_RRESP(rresp),
        .S_AXI_RVALID(rvalid),
        .S_AXI_RREADY(rready),
        .trap_vector(external_trap_vector),
        .trap_latched(trap_latched),
        .trap_status_vector(trap_status_vector),
        .staged_overflow(staged_overflow),
        .staged_underflow(staged_underflow),
        .update_pulse(update_pulse),
        .apply_pulse(apply_pulse),
        .rollback_pulse(rollback_pulse),
        .trap_clear_pulse(trap_clear_pulse),
        .checksum_mismatch_pulse(checksum_mismatch_pulse),
        .invalid_selection_pulse(invalid_selection_pulse),
        .read_only_bank_pulse(read_only_bank_pulse),
        .partial_write_pulse(partial_write_pulse),
        .shadow_loaded(shadow_loaded),
        .parameter_words(parameter_words)
    );

    task axi_write;
        input [31:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            awaddr = addr;
            wdata = data;
            awvalid = 1'b1;
            wvalid = 1'b1;
            bready = 1'b1;
            @(negedge clk);
            awvalid = 1'b0;
            wvalid = 1'b0;
            @(negedge clk);
            bready = 1'b0;
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        axi_write(32'h108, 32'd0);
        axi_write(32'h10C, 32'd0);
        axi_write(32'h110, 32'h00010000);
        axi_write(32'h114, 32'h00000000);
        axi_write(32'h120, 32'h27E798F0);
        axi_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b1 || trap_status_vector[0] !== 1'b1 || staged_overflow !== 1'b1) begin
            $display("expected generated overflow trap, status=%b overflow=%b", trap_status_vector, staged_overflow);
            $finish(1);
        end
        if (shadow_loaded !== 1'b0 || parameter_words !== 16'h0000) begin
            $display("overflowing staged update modified active or shadow state");
            $finish(1);
        end

        axi_write(32'h110, 32'h00007FFF);
        axi_write(32'h114, 32'hFFFFFFFF);
        axi_write(32'h120, 32'h0E6BCD92);
        axi_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b1 || trap_status_vector[1:0] !== 2'b11 || staged_underflow !== 1'b1) begin
            $display("expected generated underflow trap, status=%b underflow=%b", trap_status_vector, staged_underflow);
            $finish(1);
        end

        axi_write(32'h11C, 32'h00000001);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b1 || trap_status_vector !== 6'b000010) begin
            $display("selective overflow trap clear failed, status=%b", trap_status_vector);
            $finish(1);
        end

        axi_write(32'h11C, 32'h00000002);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $display("selective underflow trap clear failed, status=%b", trap_status_vector);
            $finish(1);
        end

        $finish(0);
    end
endmodule
""",
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(source_path), str(tb_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [vvp, str(sim_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
