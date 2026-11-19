# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Register-map and fail-closed bus-wrapper validation contracts

"""Verify register-map offsets and fail-closed validation of bus-wrapper inputs."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import cast

import pytest

from tests.bus_interface_wrappers_support import (
    LIF_PARAMS,
    BusProtocol,
    generate_bus_wrapper,
    generate_register_map,
)


class TestRegisterMap:
    """Test register map generation and wrapper argument validation."""

    def test_standard_layout(self) -> None:
        """Place standard control, current, spike-count, and parameter registers at 4-byte steps."""
        rmap = generate_register_map(LIF_PARAMS)
        assert rmap["CTRL"] == 0
        assert rmap["I_T"] == 4
        assert rmap["SPIKE_COUNT"] == 8
        assert rmap["P_V_REST"] == 12

    def test_custom_base_address(self) -> None:
        """Shift all register addresses by base_address."""
        rmap = generate_register_map(LIF_PARAMS, base_address=0x1000)
        assert rmap["CTRL"] == 0x1000
        assert rmap["I_T"] == 0x1004

    def test_invalid_bus(self) -> None:
        """Reject unsupported bus protocol names."""
        with pytest.raises(ValueError, match="Unsupported bus"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, bus=cast(BusProtocol, "spi"))

    def test_fail_closed_wrapper_validation(self) -> None:
        """Reject malformed identifiers, collisions, invalid widths, and narrow address buses."""
        with pytest.raises(ValueError, match="inner_module"):
            generate_bus_wrapper("bad-name", LIF_PARAMS)
        with pytest.raises(ValueError, match="parameter name"):
            generate_bus_wrapper("sc_lif", {"1bad": 8})
        with pytest.raises(ValueError, match="reserved register"):
            generate_bus_wrapper("sc_lif", {"ctrl": 8})
        with pytest.raises(ValueError, match="collides"):
            generate_bus_wrapper("sc_lif", {"tau": 8, "TAU": 16})
        with pytest.raises(ValueError, match="bus_data_width"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, bus_data_width=24)
        with pytest.raises(ValueError, match="bus_data_width"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, bus_data_width=True)
        with pytest.raises(ValueError, match="data_width"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, data_width=0)
        with pytest.raises(ValueError, match="data_width"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, data_width=64, bus_data_width=32)
        with pytest.raises(ValueError, match="width"):
            generate_bus_wrapper("sc_lif", {"P_GAIN": 0})
        with pytest.raises(ValueError, match="width"):
            generate_bus_wrapper("sc_lif", {"P_GAIN": 40}, bus_data_width=32)
        with pytest.raises(ValueError, match="addr_width"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, addr_width=0)
        with pytest.raises(ValueError, match="addr_width"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, addr_width=4)
        with pytest.raises(ValueError, match="addr_width"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, addr_width=False)
        assert "module sc_lif_axi_lite" in generate_bus_wrapper(
            "sc_lif", LIF_PARAMS, addr_width=5, bus_data_width=64
        )

    @pytest.mark.parametrize("bus", ["axi_lite", "wishbone"])
    @pytest.mark.parametrize(("bdw", "ndw"), [(8, 1), (16, 8), (32, 16), (64, 64)])
    def test_all_supported_bus_widths_match_published_register_map(
        self,
        tmp_path: Path,
        bus: BusProtocol,
        bdw: int,
        ndw: int,
    ) -> None:
        """Verify 8/16/32/64-bit wrappers lint clean and match generate_register_map in vvp."""
        assert shutil.which("verilator") is not None
        assert shutil.which("iverilog") is not None
        tmp_path.mkdir(parents=True, exist_ok=True)
        params = {"P_FULL": bdw, "P_BIT": 1}
        rmap = generate_register_map(params)
        assert rmap == {"CTRL": 0, "I_T": 4, "SPIKE_COUNT": 8, "P_FULL": 12, "P_BIT": 16}

        it_port = "input wire I_t," if ndw == 1 else f"input wire [{ndw - 1}:0] I_t,"
        full_val = (1 << (bdw - 1)) | 5
        neuron_sv = (
            "`timescale 1ns/1ps\n"
            "module sc_width_probe (\n"
            "    input wire clk,\n"
            "    input wire rst,\n"
            "    input wire en,\n"
            f"    {it_port}\n"
            f"    input wire [{bdw - 1}:0] P_FULL,\n"
            "    input wire P_BIT,\n"
            "    output reg spike_out\n"
            ");\n"
            "    always @(posedge clk) begin\n"
            "        if (rst) spike_out <= 1'b0;\n"
            f"        else spike_out <= en && P_BIT && (^I_t) && (P_FULL == {bdw}'d{full_val});\n"
            "    end\n"
            "endmodule\n"
        )
        wrapper_sv = generate_bus_wrapper(
            "sc_width_probe",
            params,
            bus=bus,
            data_width=ndw,
            addr_width=8,
            bus_data_width=bdw,
        )
        top = "sc_width_probe_axi_lite" if bus == "axi_lite" else "sc_width_probe_wb"
        neuron_path = tmp_path / "sc_width_probe.sv"
        dut_path = tmp_path / f"{top}.sv"
        neuron_path.write_text(neuron_sv, encoding="utf-8")
        dut_path.write_text(wrapper_sv, encoding="utf-8")
        lint = subprocess.run(
            [
                "verilator",
                "--lint-only",
                "-Wall",
                "--top-module",
                top,
                str(neuron_path),
                str(dut_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert lint.returncode == 0, lint.stderr

        nbytes = bdw // 8
        strb_all = (1 << nbytes) - 1
        if bus == "axi_lite":
            bus_decl_and_tasks = (
                "    reg rstn = 0;\n"
                "    reg [7:0] awaddr = 0, araddr = 0;\n"
                "    reg awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;\n"
                f"    reg [{bdw - 1}:0] wdata = 0;\n"
                f"    reg [{nbytes - 1}:0] wstrb = {nbytes}'d{strb_all};\n"
                "    wire awready, wready, bvalid, arready, rvalid, irq_spike;\n"
                "    wire [1:0] bresp, rresp;\n"
                f"    wire [{bdw - 1}:0] rdata;\n"
                f"    {top} dut (\n"
                "        .S_AXI_ACLK(clk), .S_AXI_ARESETN(rstn),\n"
                "        .S_AXI_AWADDR(awaddr), .S_AXI_AWVALID(awvalid), .S_AXI_AWREADY(awready),\n"
                "        .S_AXI_WDATA(wdata), .S_AXI_WSTRB(wstrb),\n"
                "        .S_AXI_WVALID(wvalid), .S_AXI_WREADY(wready),\n"
                "        .S_AXI_BRESP(bresp), .S_AXI_BVALID(bvalid), .S_AXI_BREADY(bready),\n"
                "        .S_AXI_ARADDR(araddr), .S_AXI_ARVALID(arvalid), .S_AXI_ARREADY(arready),\n"
                "        .S_AXI_RDATA(rdata), .S_AXI_RRESP(rresp),\n"
                "        .S_AXI_RVALID(rvalid), .S_AXI_RREADY(rready), .irq_spike(irq_spike)\n"
                "    );\n"
                f"    task automatic bus_write(input [7:0] a, input [{bdw - 1}:0] d);\n"
                "        begin\n"
                "            @(posedge clk); awaddr <= a; awvalid <= 1; wdata <= d; wvalid <= 1;\n"
                "            while (!(awvalid && awready)) @(posedge clk);\n"
                "            awvalid <= 0; wvalid <= 0;\n"
                "            while (!bvalid) @(posedge clk);\n"
                "            @(posedge clk);\n"
                "        end\n"
                "    endtask\n"
                f"    task automatic bus_read(input [7:0] a, output [{bdw - 1}:0] d);\n"
                "        begin\n"
                "            @(posedge clk); araddr <= a; arvalid <= 1;\n"
                "            while (!(arvalid && arready)) @(posedge clk);\n"
                "            arvalid <= 0;\n"
                "            while (!rvalid) @(posedge clk);\n"
                "            d = rdata;\n"
                "            @(posedge clk);\n"
                "        end\n"
                "    endtask\n"
                "    task automatic release_reset; begin #10 rstn = 1; end endtask\n"
            )
        else:
            bus_decl_and_tasks = (
                "    reg rst = 1;\n"
                "    reg [7:0] adr = 0;\n"
                f"    reg [{bdw - 1}:0] dat_i = 0;\n"
                f"    wire [{bdw - 1}:0] dat_o;\n"
                "    reg we = 0, stb = 0, cyc = 0;\n"
                "    wire ack, irq_spike;\n"
                f"    {top} dut (\n"
                "        .wb_clk_i(clk), .wb_rst_i(rst), .wb_adr_i(adr),\n"
                "        .wb_dat_i(dat_i), .wb_dat_o(dat_o), .wb_we_i(we),\n"
                "        .wb_stb_i(stb), .wb_cyc_i(cyc), .wb_ack_o(ack), .irq_spike(irq_spike)\n"
                "    );\n"
                f"    task automatic bus_write(input [7:0] a, input [{bdw - 1}:0] d);\n"
                "        begin\n"
                "            @(posedge clk); adr <= a; dat_i <= d; we <= 1; stb <= 1; cyc <= 1;\n"
                "            while (!ack) @(posedge clk);\n"
                "            stb <= 0; cyc <= 0; we <= 0;\n"
                "            @(posedge clk);\n"
                "        end\n"
                "    endtask\n"
                f"    task automatic bus_read(input [7:0] a, output [{bdw - 1}:0] d);\n"
                "        begin\n"
                "            @(posedge clk); adr <= a; we <= 0; stb <= 1; cyc <= 1;\n"
                "            while (!ack) @(posedge clk);\n"
                "            d = dat_o; stb <= 0; cyc <= 0;\n"
                "            @(posedge clk);\n"
                "        end\n"
                "    endtask\n"
                "    task automatic release_reset; begin #10 rst = 0; end endtask\n"
            )

        tb_sv = (
            "`timescale 1ns/1ps\n"
            "module tb_map;\n"
            "    reg clk = 0;\n"
            "    always #2 clk = ~clk;\n"
            f"{bus_decl_and_tasks}"
            f"    reg [{bdw - 1}:0] rd;\n"
            "    initial begin\n"
            "        release_reset();\n"
            f"        bus_write(8'd{rmap['I_T']}, {bdw}'d1);\n"
            f"        bus_write(8'd{rmap['P_FULL']}, {bdw}'d{full_val});\n"
            f"        bus_write(8'd{rmap['P_BIT'] + 1}, {bdw}'d1);\n"
            f"        bus_read(8'd{rmap['P_FULL']}, rd);\n"
            f'        if (rd !== {bdw}\'d{full_val}) $fatal(1, "P_FULL map mismatch");\n'
            f"        bus_read(8'd{rmap['P_BIT']}, rd);\n"
            f'        if (rd !== {bdw}\'d1) $fatal(1, "unaligned P_BIT word decode mismatch");\n'
            f"        bus_write(8'd{rmap['CTRL']}, {bdw}'d1);\n"
            "        repeat (3) @(posedge clk);\n"
            f"        bus_write(8'd{rmap['CTRL']}, {bdw}'d0);\n"
            f"        bus_read(8'd{rmap['SPIKE_COUNT']}, rd);\n"
            f'        if (rd < {bdw}\'d2) $fatal(1, "SPIKE_COUNT did not increment at map addr");\n'
            '        $display("MAP_DECODE_PASS");\n'
            "        $finish;\n"
            "    end\n"
            "endmodule\n"
        )
        tb_path = tmp_path / "tb_map.sv"
        vvp_path = tmp_path / "sim_map.vvp"
        tb_path.write_text(tb_sv, encoding="utf-8")
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
        assert "MAP_DECODE_PASS" in run.stdout
