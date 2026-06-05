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
    assert "output reg                          checksum_mismatch_pulse" in source
    assert "output reg                          invalid_selection_pulse" in source
    assert "localparam [31:0] UPDATE_CRC32_POLY_REFLECTED = 32'hEDB88320;" in source
    assert "function automatic [31:0] live_update_crc32;" in source
    assert (
        "wire [DATA_WIDTH-1:0] observed_checksum = live_update_crc32(reg_bank_select, reg_entry_index, reg_write_data_lo, reg_write_data_hi);"
        in source
    )
    assert "wire checksum_valid = (reg_write_checksum == observed_checksum);" in source
    assert "wire weights_staged_overflow" in source
    assert "wire weights_staged_underflow" in source
    assert "wire staged_update_fault = staged_overflow_fault | staged_underflow_fault;" in source
    assert "TRAP_CHECKSUM_MISMATCH_VECTOR" in source
    assert "TRAP_INVALID_SELECTION_VECTOR" in source
    assert "if (checksum_valid && !staged_update_fault && bank_entry_selection_valid) begin" in source
    assert "checksum_mismatch_pulse <= 1'b1;" in source
    assert "invalid_selection_pulse <= 1'b1;" in source
    assert "reg_trap_vector <= reg_trap_vector | observed_trap_vector;" in source
    assert (
        "ADDR_TRAP_STAT: S_AXI_RDATA <= {{(DATA_WIDTH-TRAP_WIDTH){1'b0}}, reg_trap_vector};"
        in source
    )
    assert "trap_clear_pulse <= 1'b1;" in source


def test_live_parameter_bank_emits_pcie_mmio_register_window_contract() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x2000,
        parameter_count=1,
        parameter_names=("w0",),
    )

    source = generate_live_parameter_bank(
        MMIOUpdateSpec(bus_protocol="pcie", banks=(bank,), control_base_address_bytes=0x100),
        module_name="sc_live_pcie_params",
    )

    assert "module sc_live_pcie_params" in source
    assert "PCIe hard IP must expose decoded single-clock MMIO read/write strobes." in source
    assert "module sc_live_pcie_params_axi4_lite_core" in source
    assert "pcie_mmio_write_valid" in source
    assert "pcie_mmio_write_response_valid" in source
    assert "pcie_mmio_read_data_valid" in source
    assert "checksum_mismatch_pulse" in source
    assert "invalid_selection_pulse" in source
    assert "assign pcie_mmio_write_ready = core_awready & core_wready;" in source
    assert ".S_AXI_AWVALID(pcie_mmio_write_valid)" in source
    assert ".S_AXI_WVALID(pcie_mmio_write_valid)" in source
    assert ".S_AXI_BREADY(1'b1)" in source
    assert ".S_AXI_RREADY(1'b1)" in source

    with pytest.raises(ValueError, match="Unsupported MMIO protocol"):
        MMIOUpdateSpec(bus_protocol="wishbone", banks=(bank,))

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


def test_live_parameter_bank_pcie_mmio_commits_shadow_update(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for PCIe MMIO live-control simulation")

    spec = MMIOUpdateSpec(
        bus_protocol="pcie",
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
    source_path = tmp_path / "sc_live_pcie_params.sv"
    tb_path = tmp_path / "tb_sc_live_pcie_params.sv"
    sim_path = tmp_path / "sc_live_pcie_params.out"
    source_path.write_text(
        generate_live_parameter_bank(spec, module_name="sc_live_pcie_params"),
        encoding="utf-8",
    )
    tb_path.write_text(
        """
module tb_sc_live_pcie_params;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [31:0] write_addr = 32'd0;
    reg [31:0] write_data = 32'd0;
    reg [3:0] write_strobe = 4'hF;
    reg write_valid = 1'b0;
    wire write_ready;
    wire write_response_valid;
    wire write_error;
    reg [31:0] read_addr = 32'd0;
    reg read_valid = 1'b0;
    wire read_ready;
    wire [31:0] read_data;
    wire read_data_valid;
    wire read_error;
    reg [3:0] external_trap_vector = 4'b000;
    wire trap_latched;
    wire [3:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    reg observed_checksum_mismatch = 1'b0;
    reg observed_invalid_selection = 1'b0;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
    wire checksum_mismatch_pulse;
    wire invalid_selection_pulse;
    wire shadow_loaded;
    wire [15:0] parameter_words;

    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (checksum_mismatch_pulse) begin
            observed_checksum_mismatch <= 1'b1;
        end
        if (invalid_selection_pulse) begin
            observed_invalid_selection <= 1'b1;
        end
    end

    sc_live_pcie_params dut (
        .pcie_clk(clk),
        .pcie_resetn(rst_n),
        .pcie_mmio_write_addr(write_addr),
        .pcie_mmio_write_data(write_data),
        .pcie_mmio_write_strobe(write_strobe),
        .pcie_mmio_write_valid(write_valid),
        .pcie_mmio_write_ready(write_ready),
        .pcie_mmio_write_response_valid(write_response_valid),
        .pcie_mmio_write_error(write_error),
        .pcie_mmio_read_addr(read_addr),
        .pcie_mmio_read_valid(read_valid),
        .pcie_mmio_read_ready(read_ready),
        .pcie_mmio_read_data(read_data),
        .pcie_mmio_read_data_valid(read_data_valid),
        .pcie_mmio_read_error(read_error),
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
        .shadow_loaded(shadow_loaded),
        .parameter_words(parameter_words)
    );

    task pcie_write;
        input [31:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            write_addr = addr;
            write_data = data;
            write_valid = 1'b1;
            @(negedge clk);
            write_valid = 1'b0;
            @(negedge clk);
            if (write_error !== 1'b0) begin
                $display("unexpected PCIe MMIO write error at %h", addr);
                $finish(1);
            end
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        pcie_write(32'h108, 32'd0);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h110, 32'h00001234);
        pcie_write(32'h114, 32'h00000000);
        pcie_write(32'h120, 32'h00001234);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words !== 16'h0000) begin
            $display("stale guard unexpectedly loaded the shadow bank");
            $finish(5);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[2] !== 1'b1 || observed_checksum_mismatch !== 1'b1) begin
            $display("stale guard did not raise checksum-mismatch trap, status=%b observed=%b", trap_status_vector, observed_checksum_mismatch);
            $finish(6);
        end

        pcie_write(32'h11C, 32'h0000000F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 4'b0000) begin
            $display("checksum-mismatch trap clear failed, status=%b", trap_status_vector);
            $finish(7);
        end

        pcie_write(32'h108, 32'd2);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h120, 32'h9ADDBE56);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words !== 16'h0000) begin
            $display("invalid bank selection unexpectedly loaded shadow state");
            $finish(8);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[3] !== 1'b1 || observed_invalid_selection !== 1'b1) begin
            $display("invalid bank selection did not raise trap, status=%b observed=%b", trap_status_vector, observed_invalid_selection);
            $finish(9);
        end

        pcie_write(32'h11C, 32'h0000000F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 4'b0000) begin
            $display("invalid-selection trap clear failed, status=%b", trap_status_vector);
            $finish(10);
        end

        pcie_write(32'h108, 32'd0);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h120, 32'h1D7D9B35);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b1 || parameter_words !== 16'h0000) begin
            $display("shadow load failed or active bank mutated early");
            $finish(2);
        end

        pcie_write(32'h100, 32'h00000002);
        repeat (2) @(negedge clk);
        if (parameter_words !== 16'h1234 || shadow_loaded !== 1'b0) begin
            $display("PCIe MMIO commit failed, parameter=%h shadow=%b", parameter_words, shadow_loaded);
            $finish(3);
        end
        if (trap_latched !== 1'b0 || trap_status_vector !== 4'b0000 || staged_overflow !== 1'b0 || staged_underflow !== 1'b0) begin
            $display("valid PCIe MMIO update raised a trap");
            $finish(4);
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
    reg [3:0] external_trap_vector = 4'b000;
    wire trap_latched;
    wire [3:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
    wire checksum_mismatch_pulse;
    wire invalid_selection_pulse;
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

        axi_write(32'h11C, 32'h00000003);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 4'b0000) begin
            $display("trap clear failed, status=%b", trap_status_vector);
            $finish(1);
        end

        axi_write(32'h110, 32'h00007FFF);
        axi_write(32'h114, 32'hFFFFFFFF);
        axi_write(32'h120, 32'h0E6BCD92);
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
