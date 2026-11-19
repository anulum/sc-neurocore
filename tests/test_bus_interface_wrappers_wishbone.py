# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wishbone B4 neuron bus-wrapper emission and co-simulation tests

"""Verify Wishbone B4 neuron wrapper emission, Verilator lint, and Icarus simulation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from tests.bus_interface_wrappers_support import LIF_PARAMS, generate_bus_wrapper
from tests.test_bus_interface_wrappers_axi4_lite import _MIXED_NEURON_SV


class TestWishbone:
    """Test Wishbone B4 wrapper generation and bus simulation."""

    def test_generates_module(self) -> None:
        """Produce a complete Wishbone B4 SystemVerilog wrapper module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "module sc_lif_wb" in v
        assert "endmodule" in v

    def test_has_wishbone_ports(self) -> None:
        """Include all Wishbone B4 slave interface signals."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        for sig in [
            "wb_clk_i",
            "wb_rst_i",
            "wb_adr_i",
            "wb_dat_i",
            "wb_dat_o",
            "wb_we_i",
            "wb_stb_i",
            "wb_cyc_i",
            "wb_ack_o",
        ]:
            assert sig in v, f"Missing Wishbone signal: {sig}"

    def test_has_interrupt_and_spike_out_binding(self) -> None:
        """Connect u_neuron.spike_out to spike_out and export irq_spike."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert ".spike_out(spike_out)" in v
        assert "assign irq_spike = spike_out;" in v

    def test_has_neuron_instance(self) -> None:
        """Instantiate the inner neuron module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "sc_lif u_neuron" in v

    def test_has_parameter_registers_and_empty_params(self) -> None:
        """Slice parameter registers by declared bit width and support empty parameter maps."""
        v = generate_bus_wrapper(
            "sc_lif_mixed",
            {"P_GAIN": 8, "P_FLAG": 1, "P_WIDE": 32},
            bus="wishbone",
            data_width=16,
            addr_width=8,
            bus_data_width=32,
        )
        assert ".P_GAIN(reg_p_gain[7:0])" in v
        assert ".P_FLAG(reg_p_flag[0])" in v
        assert ".P_WIDE(reg_p_wide)" in v
        empty_v = generate_bus_wrapper("sc_empty", {}, bus="wishbone")
        assert ".spike_out(spike_out)" in empty_v

    def test_cosim_wishbone_spike_out_and_per_param_widths(self, tmp_path: Path) -> None:
        """Verify Wishbone B4 writes, reads, spike interrupt, and spike counter."""
        assert shutil.which("iverilog") is not None
        tmp_path.mkdir(parents=True, exist_ok=True)
        wrapper_sv = generate_bus_wrapper(
            "sc_lif_mixed",
            {"P_GAIN": 8, "P_FLAG": 1, "P_WIDE": 32},
            bus="wishbone",
            data_width=16,
            addr_width=8,
            bus_data_width=32,
        )
        neuron_path = tmp_path / "sc_lif_mixed.sv"
        dut_path = tmp_path / "sc_lif_mixed_wb.sv"
        neuron_path.write_text(_MIXED_NEURON_SV, encoding="utf-8")
        dut_path.write_text(wrapper_sv, encoding="utf-8")

        assert shutil.which("verilator") is not None
        lint = subprocess.run(
            [
                "verilator",
                "--lint-only",
                "-Wall",
                "--top-module",
                "sc_lif_mixed_wb",
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
module tb_wb_wrapper;
    reg clk = 0;
    always #2 clk = ~clk;
    reg rst = 1;
    reg [7:0] adr = 0;
    reg [31:0] dat_i = 0;
    wire [31:0] dat_o;
    reg we = 0, stb = 0, cyc = 0;
    wire ack, irq_spike;

    sc_lif_mixed_wb dut (
        .wb_clk_i(clk), .wb_rst_i(rst), .wb_adr_i(adr),
        .wb_dat_i(dat_i), .wb_dat_o(dat_o), .wb_we_i(we),
        .wb_stb_i(stb), .wb_cyc_i(cyc), .wb_ack_o(ack),
        .irq_spike(irq_spike)
    );

    task automatic wb_write(input [7:0] a, input [31:0] d);
        integer t;
        begin
            @(posedge clk); adr <= a; dat_i <= d; we <= 1; stb <= 1; cyc <= 1;
            for (t = 0; t < 10 && !ack; t = t + 1) @(posedge clk);
            if (!ack) $fatal(1, "Wishbone write ACK timeout");
            stb <= 0; cyc <= 0; we <= 0;
            @(posedge clk);
        end
    endtask

    task automatic wb_read(input [7:0] a, output [31:0] d);
        integer t;
        begin
            @(posedge clk); adr <= a; we <= 0; stb <= 1; cyc <= 1;
            for (t = 0; t < 10 && !ack; t = t + 1) @(posedge clk);
            if (!ack) $fatal(1, "Wishbone read ACK timeout");
            d = dat_o;
            stb <= 0; cyc <= 0;
            @(posedge clk);
        end
    endtask

    reg [31:0] rd;
    integer irq_pulses = 0;
    always @(posedge clk) if (!rst && irq_spike === 1'b1) irq_pulses <= irq_pulses + 1;

    initial begin
        #10 rst = 0;
        wb_write(8'h04, 32'h0000_0120);
        wb_write(8'h0C, 32'h0000_005A);
        wb_write(8'h10, 32'h0000_0001);
        wb_write(8'h14, 32'hCAFE_BEBE);
        wb_write(8'h00, 32'h0000_0001);
        repeat (4) @(posedge clk);
        wb_write(8'h00, 32'h0000_0000);
        if (irq_pulses < 3)
            $fatal(1, "expected >=3 Wishbone irq_spike pulses, got %0d", irq_pulses);
        wb_read(8'h08, rd);
        if (rd !== irq_pulses) $fatal(1, "expected SPIKE_COUNT=%0d, got %0d", irq_pulses, rd);
        wb_write(8'h08, 32'h0000_0099);
        wb_write(8'hFC, 32'h1234_5678);
        wb_read(8'h08, rd);
        if (rd !== irq_pulses) $fatal(1, "read-only SPIKE_COUNT overwritten: %0d", rd);
        wb_read(8'hFC, rd);
        if (rd !== 32'd0) $fatal(1, "expected unmapped read=0, got %h", rd);
        wb_write(8'h00, 32'h0000_0002);
        wb_write(8'h00, 32'h0000_0000);
        wb_read(8'h08, rd);
        if (rd !== 32'd0) $fatal(1, "CTRL[1] did not clear Wishbone SPIKE_COUNT: %0d", rd);
        $display("WB_WRAPPER_PASS");
        $finish;
    end
endmodule
"""
        tb_path = tmp_path / "tb_wb.sv"
        vvp_path = tmp_path / "sim_wb.vvp"
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
        assert "WB_WRAPPER_PASS" in run.stdout
