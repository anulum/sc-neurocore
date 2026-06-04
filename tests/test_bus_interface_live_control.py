# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for live-control bus interface RTL generation

"""Module-specific tests for live-control parameter-bank RTL emission."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from sc_neurocore.compiler.live_control import MMIOUpdateSpec, ParameterBankSpec, TrapSpec
from sc_neurocore.hdl_gen.bus_interface import generate_live_parameter_bank


def test_live_parameter_bank_emits_bram_and_distributed_banks() -> None:
    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        control_base_address_bytes=0x100,
        banks=(
            ParameterBankSpec(
                bank_name="weights",
                start_address_bytes=0x2000,
                parameter_count=4,
                parameter_names=("w0", "w1", "w2", "w3"),
                q_format="Q8.8",
            ),
            ParameterBankSpec(
                bank_name="kuramoto",
                start_address_bytes=0x3000,
                parameter_count=128,
                parameter_names=("k_mag",),
                q_format="Q16.16",
                reset_value=-1,
            ),
        ),
        trap=TrapSpec(max_flags=8),
    )

    source = generate_live_parameter_bank(spec, module_name="sc_live_params")

    assert "module sc_live_params" in source
    assert '(* ram_style = "distributed" *) reg [15:0] weights [0:3];' in source
    assert '(* ram_style = "distributed" *) reg [15:0] shadow_weights [0:3];' in source
    assert '(* ram_style = "block" *) reg [31:0] kuramoto [0:127];' in source
    assert '(* ram_style = "block" *) reg [31:0] shadow_kuramoto [0:127];' in source
    assert "localparam [ADDR_WIDTH-1:0] ADDR_CONTROL    = 32'h100;" in source
    assert "localparam [ADDR_WIDTH-1:0] ADDR_BANK_SEL   = 32'h108;" in source
    assert "localparam [ADDR_WIDTH-1:0] ADDR_TRAP_CLEAR = 32'h11C;" in source
    assert "localparam [ADDR_WIDTH-1:0] ADDR_CHECKSUM   = 32'h120;" in source
    assert "shadow_weights[reg_entry_index] <= staged_word[15:0];" in source
    assert "shadow_kuramoto[reg_entry_index] <= staged_word[31:0];" in source
    assert "weights[reg_entry_index] <= shadow_weights[reg_entry_index];" in source
    assert "shadow_weights[reg_entry_index] <= weights[reg_entry_index];" in source
    assert "assign parameter_words[0 +: 16] = weights[0];" in source
    assert "output reg                          apply_pulse" in source
    assert "output reg                          rollback_pulse" in source
    assert "output wire                         shadow_loaded" in source
    assert "output wire [TRAP_WIDTH-1:0]        trap_status_vector" in source
    assert "output wire                         staged_overflow" in source
    assert "output wire                         staged_underflow" in source
    assert "wire checksum_valid = (reg_write_checksum == observed_checksum);" in source
    assert "wire weights_staged_overflow" in source
    assert "wire weights_staged_underflow" in source
    assert "wire staged_update_fault = staged_overflow_fault | staged_underflow_fault;" in source
    assert "if (checksum_valid && !staged_update_fault) begin" in source
    assert "reg_trap_vector <= reg_trap_vector | observed_trap_vector;" in source
    assert (
        "ADDR_TRAP_STAT: S_AXI_RDATA <= {{(DATA_WIDTH-TRAP_WIDTH){1'b0}}, reg_trap_vector};"
        in source
    )
    assert "trap_clear_pulse <= 1'b1;" in source


def test_live_parameter_bank_rejects_non_axi_and_identifier_injection() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x2000,
        parameter_count=1,
        parameter_names=("w0",),
    )

    with pytest.raises(ValueError, match="axi4_lite"):
        generate_live_parameter_bank(MMIOUpdateSpec(bus_protocol="pcie", banks=(bank,)))

    spec = MMIOUpdateSpec(bus_protocol="axi4_lite", banks=(bank,), control_base_address_bytes=0x100)
    with pytest.raises(ValueError, match="SystemVerilog identifier"):
        generate_live_parameter_bank(spec, module_name="bad;endmodule")


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
    reg [1:0] external_trap_vector = 2'b00;
    wire trap_latched;
    wire [1:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
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
        axi_write(32'h120, 32'h00010000);
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

        axi_write(32'h11C, 32'h00000003);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 2'b00) begin
            $display("trap clear failed, status=%b", trap_status_vector);
            $finish(1);
        end

        axi_write(32'h110, 32'h00007FFF);
        axi_write(32'h114, 32'hFFFFFFFF);
        axi_write(32'h120, 32'hFFFF8000);
        axi_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b1 || trap_status_vector[1] !== 1'b1 || staged_underflow !== 1'b1) begin
            $display("expected generated underflow trap, status=%b underflow=%b", trap_status_vector, staged_underflow);
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
