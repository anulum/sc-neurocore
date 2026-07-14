# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PCIe live-control RTL execution contracts

"""Compile and execute the generated PCIe MMIO live-control adapter."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from sc_neurocore.compiler.live_control import MMIOUpdateSpec, ParameterBankSpec, TrapSpec
from sc_neurocore.hdl_gen.bus_interface import generate_live_parameter_bank


def test_live_parameter_bank_pcie_mmio_commits_shadow_update(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError(
            "iverilog and vvp must be available for PCIe MMIO live-control simulation"
        )

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
            ParameterBankSpec(
                bank_name="calibration",
                start_address_bytes=0x3000,
                parameter_count=1,
                parameter_names=("c0",),
                q_format="Q8.8",
                writable=False,
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
    reg [5:0] external_trap_vector = 6'b000000;
    wire trap_latched;
    wire [5:0] trap_status_vector;
    wire staged_overflow;
    wire staged_underflow;
    reg observed_checksum_mismatch = 1'b0;
    reg observed_invalid_selection = 1'b0;
    reg observed_read_only_bank = 1'b0;
    reg observed_partial_write = 1'b0;
    wire update_pulse;
    wire apply_pulse;
    wire rollback_pulse;
    wire trap_clear_pulse;
    wire checksum_mismatch_pulse;
    wire invalid_selection_pulse;
    wire read_only_bank_pulse;
    wire partial_write_pulse;
    wire shadow_loaded;
    wire [31:0] parameter_words;

    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (checksum_mismatch_pulse) begin
            observed_checksum_mismatch <= 1'b1;
        end
        if (invalid_selection_pulse) begin
            observed_invalid_selection <= 1'b1;
        end
        if (read_only_bank_pulse) begin
            observed_read_only_bank <= 1'b1;
        end
        if (partial_write_pulse) begin
            observed_partial_write <= 1'b1;
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
        .read_only_bank_pulse(read_only_bank_pulse),
        .partial_write_pulse(partial_write_pulse),
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

    task pcie_partial_write;
        input [31:0] addr;
        input [31:0] data;
        input [3:0] strobe;
        begin
            @(negedge clk);
            write_addr = addr;
            write_data = data;
            write_strobe = strobe;
            write_valid = 1'b1;
            @(negedge clk);
            write_valid = 1'b0;
            @(negedge clk);
            write_strobe = 4'hF;
            if (write_error !== 1'b1) begin
                $display("partial PCIe MMIO write did not raise write error at %h", addr);
                $finish(14);
            end
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        pcie_partial_write(32'h110, 32'hFFFFEEEE, 4'h1);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words !== 32'h00000000) begin
            $display("partial write unexpectedly mutated live-control state");
            $finish(15);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[5] !== 1'b1 || observed_partial_write !== 1'b1) begin
            $display("partial write did not raise trap, status=%b observed=%b", trap_status_vector, observed_partial_write);
            $finish(16);
        end

        pcie_write(32'h11C, 32'h00000004);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b1 || trap_status_vector !== 6'b100000) begin
            $display("partial-write trap was cleared by an unrelated mask, status=%b", trap_status_vector);
            $finish(17);
        end

        pcie_write(32'h11C, 32'h00000020);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $display("partial-write trap clear failed, status=%b", trap_status_vector);
            $finish(17);
        end

        pcie_write(32'h108, 32'd0);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h110, 32'h00001234);
        pcie_write(32'h114, 32'h00000000);
        pcie_write(32'h120, 32'h00001234);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words !== 32'h00000000) begin
            $display("stale guard unexpectedly loaded the shadow bank");
            $finish(5);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[2] !== 1'b1 || observed_checksum_mismatch !== 1'b1) begin
            $display("stale guard did not raise checksum-mismatch trap, status=%b observed=%b", trap_status_vector, observed_checksum_mismatch);
            $finish(6);
        end

        pcie_write(32'h11C, 32'h0000003F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $display("checksum-mismatch trap clear failed, status=%b", trap_status_vector);
            $finish(7);
        end

        pcie_write(32'h108, 32'd2);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h120, 32'h9ADDBE56);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words !== 32'h00000000) begin
            $display("invalid bank selection unexpectedly loaded shadow state");
            $finish(8);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[3] !== 1'b1 || observed_invalid_selection !== 1'b1) begin
            $display("invalid bank selection did not raise trap, status=%b observed=%b", trap_status_vector, observed_invalid_selection);
            $finish(9);
        end

        pcie_write(32'h11C, 32'h0000003F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $display("invalid-selection trap clear failed, status=%b", trap_status_vector);
            $finish(10);
        end

        pcie_write(32'h108, 32'd1);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h120, 32'hB3150AA4);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b0 || parameter_words !== 32'h00000000) begin
            $display("read-only bank write unexpectedly loaded shadow state");
            $finish(11);
        end
        if (trap_latched !== 1'b1 || trap_status_vector[4] !== 1'b1 || observed_read_only_bank !== 1'b1) begin
            $display("read-only bank write did not raise trap, status=%b observed=%b", trap_status_vector, observed_read_only_bank);
            $finish(12);
        end

        pcie_write(32'h11C, 32'h0000003F);
        repeat (2) @(negedge clk);
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000) begin
            $display("read-only-bank trap clear failed, status=%b", trap_status_vector);
            $finish(13);
        end

        pcie_write(32'h108, 32'd0);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h120, 32'h1D7D9B35);
        pcie_write(32'h100, 32'h00000001);
        repeat (2) @(negedge clk);
        if (shadow_loaded !== 1'b1 || parameter_words !== 32'h00000000) begin
            $display("shadow load failed or active bank mutated early");
            $finish(2);
        end

        pcie_write(32'h108, 32'd1);
        pcie_write(32'h10C, 32'd0);
        pcie_write(32'h100, 32'h00000002);
        repeat (2) @(negedge clk);
        if (parameter_words[15:0] !== 16'h1234 || parameter_words[31:16] !== 16'h0000 || shadow_loaded !== 1'b0) begin
            $display("PCIe MMIO commit failed or retargeted after shadow load, parameter=%h shadow=%b", parameter_words, shadow_loaded);
            $finish(3);
        end
        if (trap_latched !== 1'b0 || trap_status_vector !== 6'b000000 || staged_overflow !== 1'b0 || staged_underflow !== 1'b0) begin
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
