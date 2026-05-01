# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bus interface wrapper generator (AXI4-Lite / Wishbone)

"""Generate SoC bus wrappers around compiled neuron modules.

Supports two bus protocols:

- **AXI4-Lite** — dominant in Xilinx/AMD Zynq and Intel Nios SoCs
- **Wishbone** — open-source standard (used with LiteX, RISC-V SoCs)

Each wrapper maps neuron parameters to memory-mapped registers and exports
the spike output as an interrupt line.

Usage::

    from sc_neurocore.hdl_gen.bus_interface import generate_bus_wrapper

    # AXI4-Lite wrapper for Zynq integration
    axi_verilog = generate_bus_wrapper(
        inner_module="sc_lif",
        params={"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16},
        bus="axi_lite",
        data_width=16,
    )

    # Wishbone wrapper for LiteX / RISC-V
    wb_verilog = generate_bus_wrapper(
        inner_module="sc_lif",
        params={"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16},
        bus="wishbone",
        data_width=16,
    )

Register Map
------------
Each parameter is assigned a 4-byte-aligned register. The layout is:

- ``0x00`` — Control/status (bit 0 = enable, bit 1 = reset)
- ``0x04`` — Input current (I_t)
- ``0x08`` — Spike count (read-only)
- ``0x0C+`` — One register per parameter, in declaration order
"""

from __future__ import annotations

from typing import Literal


BusProtocol = Literal["axi_lite", "wishbone"]


def generate_bus_wrapper(
    inner_module: str,
    params: dict[str, int],
    *,
    bus: BusProtocol = "axi_lite",
    data_width: int = 16,
    addr_width: int = 8,
    bus_data_width: int = 32,
    base_address: int = 0,
) -> str:
    """Generate a bus-attached wrapper around a compiled neuron module.

    Parameters
    ----------
    inner_module : str
        Name of the inner Verilog neuron module (e.g. ``"sc_lif"``).
    params : dict[str, int]
        Mapping from Verilog parameter name to bit width.
    bus : str
        Bus protocol: ``"axi_lite"`` or ``"wishbone"``.
    data_width : int
        Neuron fixed-point data width (e.g. 16 for Q8.8).
    addr_width : int
        Address bus width (default 8 → 256 bytes → 64 registers).
    bus_data_width : int
        Bus data width (default 32 for standard SoC buses).
    base_address : int
        Base address for documentation (not used in RTL).

    Returns
    -------
    str
        Complete SystemVerilog source for the bus wrapper module.
    """
    if bus == "axi_lite":
        return _generate_axi_lite(
            inner_module, params, data_width, addr_width, bus_data_width,
        )
    elif bus == "wishbone":
        return _generate_wishbone(
            inner_module, params, data_width, addr_width, bus_data_width,
        )
    else:
        raise ValueError(f"Unsupported bus protocol: {bus!r}. Use 'axi_lite' or 'wishbone'.")


def generate_register_map(
    params: dict[str, int],
    *,
    base_address: int = 0,
) -> dict[str, int]:
    """Return the register map for a neuron's parameters.

    Parameters
    ----------
    params : dict[str, int]
        Parameter names and their bit widths.
    base_address : int
        Starting address.

    Returns
    -------
    dict[str, int]
        Mapping from register name to byte address.
    """
    reg_map: dict[str, int] = {}
    offset = base_address
    reg_map["CTRL"] = offset; offset += 4
    reg_map["I_T"] = offset; offset += 4
    reg_map["SPIKE_COUNT"] = offset; offset += 4
    for name in params:
        reg_map[name] = offset
        offset += 4
    return reg_map


def _generate_axi_lite(
    inner_module: str,
    params: dict[str, int],
    data_width: int,
    addr_width: int,
    bus_data_width: int,
) -> str:
    """Generate an AXI4-Lite slave wrapper."""
    wrapper_name = f"{inner_module}_axi_lite"
    bdw = bus_data_width
    adw = addr_width
    ndw = data_width
    param_list = list(params.keys())

    lines = [
        f"// Auto-generated AXI4-Lite wrapper for {inner_module}",
        f"// SC-NeuroCore bus interface generator",
        f"// Bus: AXI4-Lite, Data: {bdw}-bit, Neuron: {ndw}-bit",
        f"",
        f"module {wrapper_name} #(",
        f"    parameter ADDR_WIDTH = {adw},",
        f"    parameter DATA_WIDTH = {bdw}",
        f") (",
        f"    // AXI4-Lite Slave Interface",
        f"    input  wire                    S_AXI_ACLK,",
        f"    input  wire                    S_AXI_ARESETN,",
        f"    // Write address",
        f"    input  wire [ADDR_WIDTH-1:0]   S_AXI_AWADDR,",
        f"    input  wire                    S_AXI_AWVALID,",
        f"    output reg                     S_AXI_AWREADY,",
        f"    // Write data",
        f"    input  wire [DATA_WIDTH-1:0]   S_AXI_WDATA,",
        f"    input  wire [DATA_WIDTH/8-1:0] S_AXI_WSTRB,",
        f"    input  wire                    S_AXI_WVALID,",
        f"    output reg                     S_AXI_WREADY,",
        f"    // Write response",
        f"    output reg  [1:0]              S_AXI_BRESP,",
        f"    output reg                     S_AXI_BVALID,",
        f"    input  wire                    S_AXI_BREADY,",
        f"    // Read address",
        f"    input  wire [ADDR_WIDTH-1:0]   S_AXI_ARADDR,",
        f"    input  wire                    S_AXI_ARVALID,",
        f"    output reg                     S_AXI_ARREADY,",
        f"    // Read data",
        f"    output reg  [DATA_WIDTH-1:0]   S_AXI_RDATA,",
        f"    output reg  [1:0]              S_AXI_RRESP,",
        f"    output reg                     S_AXI_RVALID,",
        f"    input  wire                    S_AXI_RREADY,",
        f"    // Interrupt",
        f"    output wire                    irq_spike",
        f");",
        f"",
        f"    // ── Register file ──────────────────────────────────────",
        f"    reg [{bdw-1}:0] reg_ctrl;        // 0x00: bit0=enable, bit1=reset",
        f"    reg [{bdw-1}:0] reg_i_t;         // 0x04: input current",
        f"    reg [{bdw-1}:0] reg_spike_count; // 0x08: spike counter (read-only)",
    ]

    # Parameter registers
    for i, pname in enumerate(param_list):
        lines.append(f"    reg [{bdw-1}:0] reg_{pname.lower()};{' ' * 4}// 0x{0x0C + i * 4:02X}: {pname}")

    lines.extend([
        f"",
        f"    // ── Neuron instance ───────────────────────────────────",
        f"    wire neuron_clk = S_AXI_ACLK;",
        f"    wire neuron_rst = ~S_AXI_ARESETN | reg_ctrl[1];",
        f"    wire neuron_en  = reg_ctrl[0];",
        f"    wire spike_out;",
        f"",
        f"    {inner_module} u_neuron (",
        f"        .clk(neuron_clk),",
        f"        .rst(neuron_rst),",
        f"        .en(neuron_en),",
        f"        .I_t(reg_i_t[{ndw-1}:0]),",
    ])

    for pname in param_list:
        lines.append(f"        .{pname}(reg_{pname.lower()}[{ndw-1}:0]),")

    lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma
    lines.extend([
        f"    );",
        f"",
        f"    assign irq_spike = spike_out;",
        f"",
        f"    // ── Spike counter ─────────────────────────────────────",
        f"    always @(posedge S_AXI_ACLK)",
        f"        if (!S_AXI_ARESETN || reg_ctrl[1])",
        f"            reg_spike_count <= 0;",
        f"        else if (spike_out)",
        f"            reg_spike_count <= reg_spike_count + 1;",
        f"",
        f"    // ── AXI4-Lite Write Logic ─────────────────────────────",
        f"    reg [ADDR_WIDTH-1:0] aw_addr;",
        f"",
        f"    always @(posedge S_AXI_ACLK) begin",
        f"        if (!S_AXI_ARESETN) begin",
        f"            S_AXI_AWREADY <= 1'b0;",
        f"            S_AXI_WREADY  <= 1'b0;",
        f"            S_AXI_BVALID  <= 1'b0;",
        f"            S_AXI_BRESP   <= 2'b00;",
        f"            reg_ctrl      <= 0;",
        f"            reg_i_t       <= 0;",
    ])

    for pname in param_list:
        lines.append(f"            reg_{pname.lower()} <= 0;")

    lines.extend([
        f"        end else begin",
        f"            // Address phase",
        f"            if (S_AXI_AWVALID && !S_AXI_AWREADY) begin",
        f"                S_AXI_AWREADY <= 1'b1;",
        f"                aw_addr <= S_AXI_AWADDR;",
        f"            end else",
        f"                S_AXI_AWREADY <= 1'b0;",
        f"",
        f"            // Data phase",
        f"            if (S_AXI_WVALID && !S_AXI_WREADY) begin",
        f"                S_AXI_WREADY <= 1'b1;",
        f"                case (aw_addr[{adw-1}:2])",
        f"                    {adw-2}'d0: reg_ctrl <= S_AXI_WDATA;",
        f"                    {adw-2}'d1: reg_i_t  <= S_AXI_WDATA;",
        f"                    // 0x08 = spike_count (read-only)",
    ])

    for i, pname in enumerate(param_list):
        lines.append(f"                    {adw-2}'d{i+3}: reg_{pname.lower()} <= S_AXI_WDATA;")

    lines.extend([
        f"                    default: ;",
        f"                endcase",
        f"            end else",
        f"                S_AXI_WREADY <= 1'b0;",
        f"",
        f"            // Response",
        f"            if (S_AXI_AWREADY && S_AXI_WREADY && !S_AXI_BVALID) begin",
        f"                S_AXI_BVALID <= 1'b1;",
        f"                S_AXI_BRESP  <= 2'b00;",
        f"            end else if (S_AXI_BREADY && S_AXI_BVALID)",
        f"                S_AXI_BVALID <= 1'b0;",
        f"        end",
        f"    end",
        f"",
        f"    // ── AXI4-Lite Read Logic ──────────────────────────────",
        f"    always @(posedge S_AXI_ACLK) begin",
        f"        if (!S_AXI_ARESETN) begin",
        f"            S_AXI_ARREADY <= 1'b0;",
        f"            S_AXI_RVALID  <= 1'b0;",
        f"            S_AXI_RRESP   <= 2'b00;",
        f"            S_AXI_RDATA   <= 0;",
        f"        end else begin",
        f"            if (S_AXI_ARVALID && !S_AXI_ARREADY) begin",
        f"                S_AXI_ARREADY <= 1'b1;",
        f"                case (S_AXI_ARADDR[{adw-1}:2])",
        f"                    {adw-2}'d0: S_AXI_RDATA <= reg_ctrl;",
        f"                    {adw-2}'d1: S_AXI_RDATA <= reg_i_t;",
        f"                    {adw-2}'d2: S_AXI_RDATA <= reg_spike_count;",
    ])

    for i, pname in enumerate(param_list):
        lines.append(f"                    {adw-2}'d{i+3}: S_AXI_RDATA <= reg_{pname.lower()};")

    lines.extend([
        f"                    default: S_AXI_RDATA <= 0;",
        f"                endcase",
        f"                S_AXI_RVALID <= 1'b1;",
        f"                S_AXI_RRESP  <= 2'b00;",
        f"            end else begin",
        f"                S_AXI_ARREADY <= 1'b0;",
        f"                if (S_AXI_RREADY && S_AXI_RVALID)",
        f"                    S_AXI_RVALID <= 1'b0;",
        f"            end",
        f"        end",
        f"    end",
        f"",
        f"endmodule",
        f"",
    ])

    return "\n".join(lines)


def _generate_wishbone(
    inner_module: str,
    params: dict[str, int],
    data_width: int,
    addr_width: int,
    bus_data_width: int,
) -> str:
    """Generate a Wishbone B4 pipelined slave wrapper."""
    wrapper_name = f"{inner_module}_wb"
    bdw = bus_data_width
    adw = addr_width
    ndw = data_width
    param_list = list(params.keys())

    lines = [
        f"// Auto-generated Wishbone B4 wrapper for {inner_module}",
        f"// SC-NeuroCore bus interface generator",
        f"// Bus: Wishbone B4, Data: {bdw}-bit, Neuron: {ndw}-bit",
        f"",
        f"module {wrapper_name} #(",
        f"    parameter ADDR_WIDTH = {adw},",
        f"    parameter DATA_WIDTH = {bdw}",
        f") (",
        f"    // Wishbone Slave Interface",
        f"    input  wire                    wb_clk_i,",
        f"    input  wire                    wb_rst_i,",
        f"    input  wire [ADDR_WIDTH-1:0]   wb_adr_i,",
        f"    input  wire [DATA_WIDTH-1:0]   wb_dat_i,",
        f"    output reg  [DATA_WIDTH-1:0]   wb_dat_o,",
        f"    input  wire                    wb_we_i,",
        f"    input  wire                    wb_stb_i,",
        f"    input  wire                    wb_cyc_i,",
        f"    output reg                     wb_ack_o,",
        f"    // Interrupt",
        f"    output wire                    irq_spike",
        f");",
        f"",
        f"    // ── Register file ──────────────────────────────────────",
        f"    reg [{bdw-1}:0] reg_ctrl;",
        f"    reg [{bdw-1}:0] reg_i_t;",
        f"    reg [{bdw-1}:0] reg_spike_count;",
    ]

    for i, pname in enumerate(param_list):
        lines.append(f"    reg [{bdw-1}:0] reg_{pname.lower()};")

    lines.extend([
        f"",
        f"    // ── Neuron instance ───────────────────────────────────",
        f"    wire neuron_rst = wb_rst_i | reg_ctrl[1];",
        f"    wire neuron_en  = reg_ctrl[0];",
        f"    wire spike_out;",
        f"",
        f"    {inner_module} u_neuron (",
        f"        .clk(wb_clk_i),",
        f"        .rst(neuron_rst),",
        f"        .en(neuron_en),",
        f"        .I_t(reg_i_t[{ndw-1}:0]),",
    ])

    for pname in param_list:
        lines.append(f"        .{pname}(reg_{pname.lower()}[{ndw-1}:0]),")

    lines[-1] = lines[-1].rstrip(",")
    lines.extend([
        f"    );",
        f"",
        f"    assign irq_spike = spike_out;",
        f"",
        f"    // ── Spike counter ─────────────────────────────────────",
        f"    always @(posedge wb_clk_i)",
        f"        if (wb_rst_i || reg_ctrl[1])",
        f"            reg_spike_count <= 0;",
        f"        else if (spike_out)",
        f"            reg_spike_count <= reg_spike_count + 1;",
        f"",
        f"    // ── Wishbone Bus Logic ────────────────────────────────",
        f"    wire wb_access = wb_stb_i && wb_cyc_i;",
        f"",
        f"    always @(posedge wb_clk_i) begin",
        f"        if (wb_rst_i) begin",
        f"            wb_ack_o <= 1'b0;",
        f"            wb_dat_o <= 0;",
        f"            reg_ctrl <= 0;",
        f"            reg_i_t  <= 0;",
    ])

    for pname in param_list:
        lines.append(f"            reg_{pname.lower()} <= 0;")

    lines.extend([
        f"        end else begin",
        f"            wb_ack_o <= 1'b0;",
        f"            if (wb_access && !wb_ack_o) begin",
        f"                wb_ack_o <= 1'b1;",
        f"                if (wb_we_i) begin",
        f"                    // Write",
        f"                    case (wb_adr_i[{adw-1}:2])",
        f"                        {adw-2}'d0: reg_ctrl <= wb_dat_i;",
        f"                        {adw-2}'d1: reg_i_t  <= wb_dat_i;",
    ])

    for i, pname in enumerate(param_list):
        lines.append(f"                        {adw-2}'d{i+3}: reg_{pname.lower()} <= wb_dat_i;")

    lines.extend([
        f"                        default: ;",
        f"                    endcase",
        f"                end else begin",
        f"                    // Read",
        f"                    case (wb_adr_i[{adw-1}:2])",
        f"                        {adw-2}'d0: wb_dat_o <= reg_ctrl;",
        f"                        {adw-2}'d1: wb_dat_o <= reg_i_t;",
        f"                        {adw-2}'d2: wb_dat_o <= reg_spike_count;",
    ])

    for i, pname in enumerate(param_list):
        lines.append(f"                        {adw-2}'d{i+3}: wb_dat_o <= reg_{pname.lower()};")

    lines.extend([
        f"                        default: wb_dat_o <= 0;",
        f"                    endcase",
        f"                end",
        f"            end",
        f"        end",
        f"    end",
        f"",
        f"endmodule",
        f"",
    ])

    return "\n".join(lines)
