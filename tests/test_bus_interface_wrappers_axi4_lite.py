# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AXI4-Lite neuron bus-wrapper emission and co-simulation tests

"""Verify AXI4-Lite neuron wrapper emission, Verilator lint, and Icarus simulation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from tests.bus_interface_wrappers_support import LIF_PARAMS, generate_bus_wrapper


_MIXED_NEURON_SV = """\
`timescale 1ns/1ps
module sc_lif_mixed (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire [15:0] I_t,
    input  wire [7:0]  P_GAIN,
    input  wire        P_FLAG,
    input  wire [31:0] P_WIDE,
    output reg         spike_out
);
    reg [1:0] phase;
    always @(posedge clk) begin
        if (rst) begin
            phase     <= 2'd0;
            spike_out <= 1'b0;
        end else if (
            en && P_FLAG && I_t == 16'h0120 && P_GAIN == 8'h5A && P_WIDE == 32'hCAFEBEBE
        ) begin
            phase     <= phase + 2'd1;
            spike_out <= (phase != 2'd3);
        end else begin
            spike_out <= 1'b0;
        end
    end
endmodule
"""


class TestAXI4Lite:
    """Test AXI4-Lite wrapper generation and cycle-accurate bus semantics."""

    def test_generates_module(self) -> None:
        """Produce a complete AXI4-Lite SystemVerilog wrapper module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "module sc_lif_axi_lite" in v
        assert "endmodule" in v

    def test_has_axi_ports(self) -> None:
        """Include all standard AXI4-Lite slave interface ports."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        for sig in [
            "S_AXI_ACLK",
            "S_AXI_ARESETN",
            "S_AXI_AWADDR",
            "S_AXI_WDATA",
            "S_AXI_WSTRB",
            "S_AXI_RDATA",
            "S_AXI_BRESP",
        ]:
            assert sig in v, f"Missing AXI signal: {sig}"

    def test_has_interrupt_and_spike_out_binding(self) -> None:
        """Connect u_neuron.spike_out to spike_out and export irq_spike."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert ".spike_out(spike_out)" in v
        assert "assign irq_spike = spike_out;" in v

    def test_has_neuron_instance(self) -> None:
        """Instantiate the inner neuron module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "sc_lif u_neuron" in v

    def test_has_parameter_registers_and_per_param_widths(self) -> None:
        """Slice each parameter register to its declared bit width."""
        v = generate_bus_wrapper(
            "sc_lif_mixed",
            {"P_GAIN": 8, "P_FLAG": 1, "P_WIDE": 32},
            bus="axi_lite",
            data_width=16,
            addr_width=8,
            bus_data_width=32,
        )
        assert ".P_GAIN(reg_p_gain[7:0])" in v
        assert ".P_FLAG(reg_p_flag[0])" in v
        assert ".P_WIDE(reg_p_wide)" in v

    def test_has_spike_counter(self) -> None:
        """Include a read-only spike counter register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "reg_spike_count" in v

    def test_control_register(self) -> None:
        """Expose enable and synchronous soft-reset bits in reg_ctrl."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "reg_ctrl" in v
        assert "reg_ctrl[0]" in v
        assert "reg_ctrl[1]" in v

    def test_cosim_spike_out_decoupled_writes_wstrb_and_backpressure(self, tmp_path: Path) -> None:
        """Verify AW+W, AW->W, W->AW, WSTRB, BREADY/RREADY stalls, and spike counting."""
        assert shutil.which("iverilog") is not None
        tmp_path.mkdir(parents=True, exist_ok=True)
        wrapper_sv = generate_bus_wrapper(
            "sc_lif_mixed",
            {"P_GAIN": 8, "P_FLAG": 1, "P_WIDE": 32},
            bus="axi_lite",
            data_width=16,
            addr_width=8,
            bus_data_width=32,
        )
        neuron_path = tmp_path / "sc_lif_mixed.sv"
        dut_path = tmp_path / "sc_lif_mixed_axi_lite.sv"
        neuron_path.write_text(_MIXED_NEURON_SV, encoding="utf-8")
        dut_path.write_text(wrapper_sv, encoding="utf-8")

        assert shutil.which("verilator") is not None
        lint = subprocess.run(
            [
                "verilator",
                "--lint-only",
                "-Wall",
                "--top-module",
                "sc_lif_mixed_axi_lite",
                str(neuron_path),
                str(dut_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert lint.returncode == 0, lint.stderr

        tb = """\
`timescale 1ns/1ps
module tb_axi_wrapper;
    reg clk = 0;
    always #2 clk = ~clk;
    reg rstn = 0;

    reg [7:0]  awaddr = 0; reg awvalid = 0; wire awready;
    reg [31:0] wdata = 0;  reg [3:0] wstrb = 4'hF; reg wvalid = 0; wire wready;
    wire [1:0] bresp; wire bvalid; reg bready = 1;
    reg [7:0]  araddr = 0; reg arvalid = 0; wire arready;
    wire [31:0] rdata; wire [1:0] rresp; wire rvalid; reg rready = 1;
    wire irq_spike;

    sc_lif_mixed_axi_lite dut (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(rstn),
        .S_AXI_AWADDR(awaddr), .S_AXI_AWVALID(awvalid), .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata), .S_AXI_WSTRB(wstrb), .S_AXI_WVALID(wvalid), .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp), .S_AXI_BVALID(bvalid), .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr), .S_AXI_ARVALID(arvalid), .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata), .S_AXI_RRESP(rresp), .S_AXI_RVALID(rvalid), .S_AXI_RREADY(rready),
        .irq_spike(irq_spike)
    );

    task automatic axi_write_simultaneous(input [7:0] addr, input [31:0] val, input [3:0] strb);
        integer t;
        reg aw_done, w_done;
        begin
            aw_done = 0; w_done = 0;
            @(posedge clk);
            awaddr <= addr; awvalid <= 1; wdata <= val; wstrb <= strb; wvalid <= 1;
            for (t = 0; t < 16 && !(aw_done && w_done); t = t + 1) begin
                @(posedge clk);
                if (awvalid && awready) begin awvalid <= 0; aw_done = 1; end
                if (wvalid && wready) begin wvalid <= 0; w_done = 1; end
            end
            if (!(aw_done && w_done)) $fatal(1, "simultaneous AW/W timeout");
            for (t = 0; t < 16 && !bvalid; t = t + 1) @(posedge clk);
            if (!bvalid || bresp !== 2'b00) $fatal(1, "simultaneous BVALID timeout");
            @(posedge clk);
        end
    endtask

    task automatic axi_write_aw_then_w(input [7:0] addr, input [31:0] val);
        integer t;
        begin
            @(posedge clk); awaddr <= addr; awvalid <= 1;
            for (t = 0; t < 12 && !(awvalid && awready); t = t + 1) @(posedge clk);
            if (!(awvalid && awready)) $fatal(1, "AW-first AWREADY timeout");
            awvalid <= 0; awaddr <= 8'hFC;
            repeat (3) @(posedge clk);
            wdata <= val; wstrb <= 4'hF; wvalid <= 1;
            for (t = 0; t < 12 && !(wvalid && wready); t = t + 1) @(posedge clk);
            if (!(wvalid && wready)) $fatal(1, "AW-first WREADY timeout");
            wvalid <= 0;
            for (t = 0; t < 12 && !bvalid; t = t + 1) @(posedge clk);
            if (!bvalid || bresp !== 2'b00) $fatal(1, "AW-first BVALID missing");
            @(posedge clk);
        end
    endtask

    task automatic axi_write_w_then_aw(input [7:0] addr, input [31:0] val);
        integer t;
        begin
            @(posedge clk); wdata <= val; wstrb <= 4'hF; wvalid <= 1;
            for (t = 0; t < 12 && !(wvalid && wready); t = t + 1) @(posedge clk);
            if (!(wvalid && wready)) $fatal(1, "W-first WREADY timeout");
            wvalid <= 0; wdata <= 32'hDEAD_DEAD;
            repeat (3) @(posedge clk);
            awaddr <= addr; awvalid <= 1;
            for (t = 0; t < 12 && !(awvalid && awready); t = t + 1) @(posedge clk);
            if (!(awvalid && awready)) $fatal(1, "W-first AWREADY timeout");
            awvalid <= 0;
            for (t = 0; t < 12 && !bvalid; t = t + 1) @(posedge clk);
            if (!bvalid || bresp !== 2'b00) $fatal(1, "W-first BVALID missing");
            @(posedge clk);
        end
    endtask

    task automatic axi_read(input [7:0] addr, output [31:0] out_val);
        integer t;
        begin
            @(posedge clk); araddr <= addr; arvalid <= 1;
            for (t = 0; t < 12 && !(arvalid && arready); t = t + 1) begin
                if (rvalid) $fatal(1, "RVALID asserted before AR handshake");
                @(posedge clk);
            end
            if (!(arvalid && arready)) $fatal(1, "ARREADY timeout");
            if (rvalid) $fatal(1, "RVALID prematurely high on AR handshake edge");
            arvalid <= 0; araddr <= 8'hFC;
            for (t = 0; t < 12 && !rvalid; t = t + 1) @(posedge clk);
            if (!rvalid || rresp !== 2'b00) $fatal(1, "RVALID timeout");
            out_val = rdata;
            @(posedge clk);
        end
    endtask

    reg [31:0] rd;
    integer irq_pulses = 0;
    integer t_stall;
    always @(posedge clk) if (rstn && irq_spike === 1'b1) irq_pulses <= irq_pulses + 1;

    initial begin
        #10 rstn = 1;
        // 1. Simultaneous write to P_WIDE (0x14), then partial byte-lane WSTRB writes
        axi_write_simultaneous(8'h14, 32'hCAFE_BABE, 4'b1111);
        axi_write_simultaneous(8'h14, 32'h0000_BEEF, 4'b0010);
        axi_write_simultaneous(8'h14, 32'hFFFF_FFFF, 4'b0000);
        axi_read(8'h14, rd);
        if (rd !== 32'hCAFE_BEBE) $fatal(1, "WSTRB mismatch: got %h", rd);

        // 2. Decoupled AW->W write to I_T (0x04) and W->AW write to P_GAIN (0x0C)
        axi_write_aw_then_w(8'h04, 32'h0000_0120);
        axi_write_w_then_aw(8'h0C, 32'h0000_005A);
        axi_read(8'h04, rd);
        if (rd !== 32'h0000_0120) $fatal(1, "I_T clobbered by W-first write: %h", rd);
        axi_read(8'h0C, rd);
        if (rd !== 32'h0000_005A) $fatal(1, "P_GAIN W-first mismatch: %h", rd);

        // 3. Stalled BREADY=0 write to P_FLAG (0x10 = 1)
        bready = 0;
        @(posedge clk); awaddr <= 8'h10; awvalid <= 1; wdata <= 32'h1; wstrb <= 4'hF; wvalid <= 1;
        @(posedge clk); @(posedge clk); awvalid <= 0; wvalid <= 0;
        repeat (3) begin
            @(posedge clk);
            if (!bvalid) $fatal(1, "BVALID dropped during BREADY=0 stall");
        end
        bready = 1;
        @(posedge clk);

        // 4. Enable neuron via CTRL (0x00 = 1), count spikes on irq_spike and SPIKE_COUNT (0x08)
        axi_write_simultaneous(8'h00, 32'h0000_0001, 4'hF);
        repeat (4) @(posedge clk);
        axi_write_simultaneous(8'h00, 32'h0000_0000, 4'hF);
        if (irq_pulses < 3) $fatal(1, "expected >=3 irq_spike pulses, got %0d", irq_pulses);

        // 5. Read SPIKE_COUNT (0x08) while stalling RREADY=0 and presenting competing ARVALID
        rready = 0;
        @(posedge clk); araddr <= 8'h08; arvalid <= 1;
        for (t_stall = 0; t_stall < 12 && !(arvalid && arready); t_stall = t_stall + 1)
            @(posedge clk);
        if (!(arvalid && arready)) $fatal(1, "stalled read ARREADY timeout");
        arvalid <= 0;
        @(posedge clk);
        araddr <= 8'h00; arvalid <= 1; // competing ARADDR while RVALID stalled!
        repeat (3) begin
            @(posedge clk);
            if (arready) $fatal(1, "ARREADY asserted while RVALID stalled");
            if (!rvalid || rdata !== irq_pulses)
                $fatal(1, "RDATA corrupted under RREADY=0 stall: %0d vs %0d", rdata, irq_pulses);
        end
        arvalid <= 0; rready <= 1;
        @(posedge clk);

        // 6. Write to read-only SPIKE_COUNT (0x08) or unmapped (0xFC) is ignored; CTRL[1] clears it
        axi_write_simultaneous(8'h08, 32'h0000_0099, 4'hF);
        axi_write_simultaneous(8'hFC, 32'h1234_5678, 4'hF);
        axi_read(8'hFC, rd);
        if (rd !== 32'd0) $fatal(1, "unmapped read non-zero: %h", rd);
        axi_read(8'h08, rd);
        if (rd !== irq_pulses) $fatal(1, "read-only SPIKE_COUNT overwritten: %0d", rd);
        axi_write_simultaneous(8'h00, 32'h0000_0002, 4'hF);
        axi_write_simultaneous(8'h00, 32'h0000_0000, 4'hF);
        axi_read(8'h08, rd);
        if (rd !== 32'd0) $fatal(1, "CTRL[1] did not reset SPIKE_COUNT: %0d", rd);

        $display("AXI_WRAPPER_PASS");
        $finish;
    end
endmodule
"""
        tb_path = tmp_path / "tb.sv"
        vvp_path = tmp_path / "sim.vvp"
        tb_path.write_text(tb, encoding="utf-8")
        comp = subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                str(vvp_path),
                str(neuron_path),
                str(dut_path),
                str(tb_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert comp.returncode == 0, comp.stderr
        run = subprocess.run(
            ["vvp", "-n", str(vvp_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
        assert "AXI_WRAPPER_PASS" in run.stdout
