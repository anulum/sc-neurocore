# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AXI4-Lite and Wishbone B4 neuron bus-wrapper renderers

"""Render protocol-specific wrappers around compiled neuron modules."""

from __future__ import annotations

from numbers import Integral

from ._ident import sanitize_ident

_RESERVED_REGISTERS = frozenset(("ctrl", "i_t", "spike_count"))
_SUPPORTED_BUS_WIDTHS = frozenset((8, 16, 32, 64))


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_wrapper_args(
    inner_module: str,
    params: dict[str, int],
    data_width: int,
    addr_width: int,
    bus_data_width: int,
) -> tuple[str, list[tuple[str, int]], int, int, int, int]:
    safe_inner = sanitize_ident(inner_module, context="inner_module")
    bdw = _require_int(bus_data_width, "bus_data_width")
    if bdw not in _SUPPORTED_BUS_WIDTHS:
        raise ValueError("bus_data_width must be one of 8, 16, 32, or 64")
    ndw = _require_int(data_width, "data_width")
    if not 1 <= ndw <= bdw:
        raise ValueError(f"data_width must be between 1 and bus_data_width ({bdw})")
    validated_params: list[tuple[str, int]] = []
    seen_regs: set[str] = set()
    for raw_name, raw_width in params.items():
        pname = sanitize_ident(raw_name, context="parameter name")
        low = pname.lower()
        if low in _RESERVED_REGISTERS:
            raise ValueError(f"parameter name {pname!r} collides with reserved register reg_{low}")
        if low in seen_regs:
            raise ValueError(f"parameter name {pname!r} collides case-insensitively with reg_{low}")
        seen_regs.add(low)
        pwidth = _require_int(raw_width, f"parameter {pname!r} width")
        if not 1 <= pwidth <= bdw:
            raise ValueError(
                f"parameter {pname!r} width must be between 1 and bus_data_width ({bdw})"
            )
        validated_params.append((pname, pwidth))
    adw = _require_int(addr_width, "addr_width")
    min_bytes = (3 + len(validated_params)) * 4
    if adw < 1 or (1 << adw) < min_bytes:
        raise ValueError(
            f"addr_width={adw} cannot address {min_bytes} register bytes; increase addr_width"
        )
    return safe_inner, validated_params, ndw, adw, bdw, adw - 2


def _slice_reg(reg_name: str, width: int, bus_data_width: int) -> str:
    if width == bus_data_width:
        return reg_name
    if width == 1:
        return f"{reg_name}[0]"
    return f"{reg_name}[{width - 1}:0]"


def render_axi_lite_wrapper(
    inner_module: str,
    params: dict[str, int],
    data_width: int,
    addr_width: int,
    bus_data_width: int,
) -> str:
    """Render an AXI4-Lite slave wrapper around a neuron module.

    Parameters
    ----------
    inner_module : str
        Name of the generated neuron module to instantiate.
    params : dict[str, int]
        Parameter-port names mapped to their bit widths.
    data_width : int
        Neuron datapath width.
    addr_width : int
        AXI address width.
    bus_data_width : int
        AXI register data width (8, 16, 32, or 64).

    Returns
    -------
    str
        Complete SystemVerilog wrapper source.
    """
    safe_inner, param_items, ndw, adw, bdw, wadw = _validate_wrapper_args(
        inner_module, params, data_width, addr_width, bus_data_width
    )
    param_regs = [
        f"    reg [DATA_WIDTH-1:0] reg_{p.lower()}; // 0x{(3 + idx) * 4:02X}: {p} ({pw}-bit)"
        for idx, (p, pw) in enumerate(param_items)
    ]
    param_ports = [
        f"        .{p}({_slice_reg(f'reg_{p.lower()}', pw, bdw)})," for p, pw in param_items
    ]
    param_resets = [
        f"            reg_{p.lower()} <= {{DATA_WIDTH{{1'b0}}}};" for p, _ in param_items
    ]
    param_writes = [
        f"                    {wadw}'d{3 + idx}: "
        f"reg_{p.lower()} <= apply_wstrb(reg_{p.lower()}, eff_w_data, eff_w_strb);"
        for idx, (p, _) in enumerate(param_items)
    ]
    param_reads = [
        f"                    {wadw}'d{3 + idx}: S_AXI_RDATA <= reg_{p.lower()};"
        for idx, (p, _) in enumerate(param_items)
    ]
    i_t_expr = _slice_reg("reg_i_t", ndw, bdw)

    lines = [
        f"// Auto-generated AXI4-Lite wrapper for {safe_inner}",
        "// SC-NeuroCore bus interface generator",
        f"// Bus: AXI4-Lite, Data: {bdw}-bit, Neuron: {ndw}-bit",
        "",
        f"module {safe_inner}_axi_lite #(",
        f"    parameter ADDR_WIDTH = {adw},",
        f"    parameter DATA_WIDTH = {bdw}",
        ") (",
        "    input  wire                    S_AXI_ACLK,",
        "    input  wire                    S_AXI_ARESETN,",
        "    input  wire [ADDR_WIDTH-1:0]   S_AXI_AWADDR,",
        "    input  wire                    S_AXI_AWVALID,",
        "    output reg                     S_AXI_AWREADY,",
        "    input  wire [DATA_WIDTH-1:0]   S_AXI_WDATA,",
        "    input  wire [DATA_WIDTH/8-1:0] S_AXI_WSTRB,",
        "    input  wire                    S_AXI_WVALID,",
        "    output reg                     S_AXI_WREADY,",
        "    output reg  [1:0]              S_AXI_BRESP,",
        "    output reg                     S_AXI_BVALID,",
        "    input  wire                    S_AXI_BREADY,",
        "    input  wire [ADDR_WIDTH-1:0]   S_AXI_ARADDR,",
        "    input  wire                    S_AXI_ARVALID,",
        "    output reg                     S_AXI_ARREADY,",
        "    output reg  [DATA_WIDTH-1:0]   S_AXI_RDATA,",
        "    output reg  [1:0]              S_AXI_RRESP,",
        "    output reg                     S_AXI_RVALID,",
        "    input  wire                    S_AXI_RREADY,",
        "    output wire                    irq_spike",
        ");",
        "",
        "    reg [DATA_WIDTH-1:0] reg_ctrl;        // bit0=enable, bit1=reset",
        "    reg [DATA_WIDTH-1:0] reg_i_t;         // input current",
        "    reg [DATA_WIDTH-1:0] reg_spike_count; // spike counter (read-only)",
        *param_regs,
        "",
        "    wire neuron_clk = S_AXI_ACLK;",
        "    wire neuron_rst = ~S_AXI_ARESETN | reg_ctrl[1];",
        "    wire neuron_en  = reg_ctrl[0];",
        "    wire spike_out;",
        "",
        f"    {safe_inner} u_neuron (",
        "        .clk(neuron_clk),",
        "        .rst(neuron_rst),",
        "        .en(neuron_en),",
        f"        .I_t({i_t_expr}),",
        *param_ports,
        "        .spike_out(spike_out)",
        "    );",
        "",
        "    assign irq_spike = spike_out;",
        "",
        "    always @(posedge S_AXI_ACLK) begin",
        "        if (!S_AXI_ARESETN || reg_ctrl[1])",
        "            reg_spike_count <= {DATA_WIDTH{1'b0}};",
        "        else if (spike_out)",
        "            reg_spike_count <= reg_spike_count + {{(DATA_WIDTH-1){1'b0}}, 1'b1};",
        "    end",
        "",
        "    function automatic [DATA_WIDTH-1:0] apply_wstrb(",
        "        input [DATA_WIDTH-1:0]   cur,",
        "        input [DATA_WIDTH-1:0]   wdat,",
        "        input [DATA_WIDTH/8-1:0] wstrb",
        "    );",
        "        integer b;",
        "        begin",
        "            apply_wstrb = cur;",
        "            for (b = 0; b < DATA_WIDTH / 8; b = b + 1)",
        "                if (wstrb[b])",
        "                    apply_wstrb[b*8 +: 8] = wdat[b*8 +: 8];",
        "        end",
        "    endfunction",
        "",
        "    wire _unused_addr_lsbs = &{1'b0, S_AXI_AWADDR[1:0], S_AXI_ARADDR[1:0], 1'b0};",
        "",
        "    reg                    aw_captured;",
        "    reg [ADDR_WIDTH-3:0]   aw_word_reg;",
        "    reg                    w_captured;",
        "    reg [DATA_WIDTH-1:0]   w_data_reg;",
        "    reg [DATA_WIDTH/8-1:0] w_strb_reg;",
        "",
        "    wire aw_hs = S_AXI_AWVALID && S_AXI_AWREADY;",
        "    wire w_hs  = S_AXI_WVALID  && S_AXI_WREADY;",
        "    wire aw_can_accept = !S_AXI_AWREADY && !aw_captured && !S_AXI_BVALID;",
        "    wire w_can_accept  = !S_AXI_WREADY  && !w_captured  && !S_AXI_BVALID;",
        "    wire wr_commit = (aw_hs && w_hs) || (aw_hs && w_captured) || (w_hs && aw_captured);",
        "    wire [ADDR_WIDTH-3:0]   aw_word_in  = S_AXI_AWADDR[ADDR_WIDTH-1:2];",
        "    wire [ADDR_WIDTH-3:0]   eff_aw_word = aw_hs ? aw_word_in : aw_word_reg;",
        "    wire [DATA_WIDTH-1:0]   eff_w_data  = w_hs  ? S_AXI_WDATA  : w_data_reg;",
        "    wire [DATA_WIDTH/8-1:0] eff_w_strb  = w_hs  ? S_AXI_WSTRB  : w_strb_reg;",
        "",
        "    always @(posedge S_AXI_ACLK) begin",
        "        if (!S_AXI_ARESETN) begin",
        "            S_AXI_AWREADY <= 1'b0;",
        "            S_AXI_WREADY  <= 1'b0;",
        "            S_AXI_BVALID  <= 1'b0;",
        "            S_AXI_BRESP   <= 2'b00;",
        "            aw_captured   <= 1'b0;",
        "            aw_word_reg   <= {(ADDR_WIDTH-2){1'b0}};",
        "            w_captured    <= 1'b0;",
        "            w_data_reg    <= {DATA_WIDTH{1'b0}};",
        "            w_strb_reg    <= {(DATA_WIDTH/8){1'b0}};",
        "            reg_ctrl      <= {DATA_WIDTH{1'b0}};",
        "            reg_i_t       <= {DATA_WIDTH{1'b0}};",
        *param_resets,
        "        end else begin",
        "            S_AXI_AWREADY <= S_AXI_AWVALID && aw_can_accept;",
        "            S_AXI_WREADY  <= S_AXI_WVALID  && w_can_accept;",
        "            if (aw_hs && !wr_commit) begin",
        "                aw_captured <= 1'b1;",
        "                aw_word_reg <= S_AXI_AWADDR[ADDR_WIDTH-1:2];",
        "            end",
        "            if (w_hs && !wr_commit) begin",
        "                w_captured <= 1'b1;",
        "                w_data_reg <= S_AXI_WDATA;",
        "                w_strb_reg <= S_AXI_WSTRB;",
        "            end",
        "            if (wr_commit) begin",
        "                aw_captured  <= 1'b0;",
        "                w_captured   <= 1'b0;",
        "                S_AXI_BVALID <= 1'b1;",
        "                S_AXI_BRESP  <= 2'b00;",
        "                case (eff_aw_word)",
        f"                    {wadw}'d0:",
        "                        reg_ctrl <= apply_wstrb(reg_ctrl, eff_w_data, eff_w_strb);",
        f"                    {wadw}'d1:",
        "                        reg_i_t <= apply_wstrb(reg_i_t, eff_w_data, eff_w_strb);",
        *param_writes,
        "                    default: ;",
        "                endcase",
        "            end else if (S_AXI_BREADY && S_AXI_BVALID) begin",
        "                S_AXI_BVALID <= 1'b0;",
        "            end",
        "        end",
        "    end",
        "",
        "    wire ar_hs = S_AXI_ARVALID && S_AXI_ARREADY;",
        "",
        "    always @(posedge S_AXI_ACLK) begin",
        "        if (!S_AXI_ARESETN) begin",
        "            S_AXI_ARREADY <= 1'b0;",
        "            S_AXI_RVALID  <= 1'b0;",
        "            S_AXI_RRESP   <= 2'b00;",
        "            S_AXI_RDATA   <= {DATA_WIDTH{1'b0}};",
        "        end else begin",
        "            S_AXI_ARREADY <= S_AXI_ARVALID && !S_AXI_ARREADY && !S_AXI_RVALID;",
        "            if (ar_hs) begin",
        "                S_AXI_RVALID <= 1'b1;",
        "                S_AXI_RRESP  <= 2'b00;",
        "                case (S_AXI_ARADDR[ADDR_WIDTH-1:2])",
        f"                    {wadw}'d0: S_AXI_RDATA <= reg_ctrl;",
        f"                    {wadw}'d1: S_AXI_RDATA <= reg_i_t;",
        f"                    {wadw}'d2: S_AXI_RDATA <= reg_spike_count;",
        *param_reads,
        "                    default: S_AXI_RDATA <= {DATA_WIDTH{1'b0}};",
        "                endcase",
        "            end else if (S_AXI_RREADY && S_AXI_RVALID) begin",
        "                S_AXI_RVALID <= 1'b0;",
        "            end",
        "        end",
        "    end",
        "",
        "endmodule",
        "",
    ]
    return "\n".join(lines)


def render_wishbone_wrapper(
    inner_module: str,
    params: dict[str, int],
    data_width: int,
    addr_width: int,
    bus_data_width: int,
) -> str:
    """Render a Wishbone B4 slave wrapper around a neuron module.

    Parameters
    ----------
    inner_module : str
        Name of the generated neuron module to instantiate.
    params : dict[str, int]
        Parameter-port names mapped to their bit widths.
    data_width : int
        Neuron datapath width.
    addr_width : int
        Wishbone address width.
    bus_data_width : int
        Wishbone register data width (8, 16, 32, or 64).

    Returns
    -------
    str
        Complete SystemVerilog wrapper source.
    """
    safe_inner, param_items, ndw, adw, bdw, wadw = _validate_wrapper_args(
        inner_module, params, data_width, addr_width, bus_data_width
    )
    param_regs = [f"    reg [DATA_WIDTH-1:0] reg_{p.lower()};" for p, _ in param_items]
    param_ports = [
        f"        .{p}({_slice_reg(f'reg_{p.lower()}', pw, bdw)})," for p, pw in param_items
    ]
    param_resets = [
        f"            reg_{p.lower()} <= {{DATA_WIDTH{{1'b0}}}};" for p, _ in param_items
    ]
    param_writes = [
        f"                        {wadw}'d{3 + idx}: reg_{p.lower()} <= wb_dat_i;"
        for idx, (p, _) in enumerate(param_items)
    ]
    param_reads = [
        f"                        {wadw}'d{3 + idx}: wb_dat_o <= reg_{p.lower()};"
        for idx, (p, _) in enumerate(param_items)
    ]
    i_t_expr = _slice_reg("reg_i_t", ndw, bdw)

    lines = [
        f"// Auto-generated Wishbone B4 wrapper for {safe_inner}",
        "// SC-NeuroCore bus interface generator",
        f"// Bus: Wishbone B4, Data: {bdw}-bit, Neuron: {ndw}-bit",
        "",
        f"module {safe_inner}_wb #(",
        f"    parameter ADDR_WIDTH = {adw},",
        f"    parameter DATA_WIDTH = {bdw}",
        ") (",
        "    input  wire                  wb_clk_i,",
        "    input  wire                  wb_rst_i,",
        "    input  wire [ADDR_WIDTH-1:0] wb_adr_i,",
        "    input  wire [DATA_WIDTH-1:0] wb_dat_i,",
        "    output reg  [DATA_WIDTH-1:0] wb_dat_o,",
        "    input  wire                  wb_we_i,",
        "    input  wire                  wb_stb_i,",
        "    input  wire                  wb_cyc_i,",
        "    output reg                   wb_ack_o,",
        "    output wire                  irq_spike",
        ");",
        "",
        "    reg [DATA_WIDTH-1:0] reg_ctrl;",
        "    reg [DATA_WIDTH-1:0] reg_i_t;",
        "    reg [DATA_WIDTH-1:0] reg_spike_count;",
        *param_regs,
        "",
        "    wire neuron_rst = wb_rst_i | reg_ctrl[1];",
        "    wire neuron_en  = reg_ctrl[0];",
        "    wire spike_out;",
        "",
        f"    {safe_inner} u_neuron (",
        "        .clk(wb_clk_i),",
        "        .rst(neuron_rst),",
        "        .en(neuron_en),",
        f"        .I_t({i_t_expr}),",
        *param_ports,
        "        .spike_out(spike_out)",
        "    );",
        "",
        "    assign irq_spike = spike_out;",
        "",
        "    always @(posedge wb_clk_i) begin",
        "        if (wb_rst_i || reg_ctrl[1])",
        "            reg_spike_count <= {DATA_WIDTH{1'b0}};",
        "        else if (spike_out)",
        "            reg_spike_count <= reg_spike_count + {{(DATA_WIDTH-1){1'b0}}, 1'b1};",
        "    end",
        "",
        "    wire _unused_addr_lsbs = &{1'b0, wb_adr_i[1:0], 1'b0};",
        "    wire wb_access = wb_stb_i && wb_cyc_i;",
        "",
        "    always @(posedge wb_clk_i) begin",
        "        if (wb_rst_i) begin",
        "            wb_ack_o <= 1'b0;",
        "            wb_dat_o <= {DATA_WIDTH{1'b0}};",
        "            reg_ctrl <= {DATA_WIDTH{1'b0}};",
        "            reg_i_t  <= {DATA_WIDTH{1'b0}};",
        *param_resets,
        "        end else begin",
        "            wb_ack_o <= 1'b0;",
        "            if (wb_access && !wb_ack_o) begin",
        "                wb_ack_o <= 1'b1;",
        "                if (wb_we_i) begin",
        "                    case (wb_adr_i[ADDR_WIDTH-1:2])",
        f"                        {wadw}'d0: reg_ctrl <= wb_dat_i;",
        f"                        {wadw}'d1: reg_i_t <= wb_dat_i;",
        *param_writes,
        "                        default: ;",
        "                    endcase",
        "                end else begin",
        "                    case (wb_adr_i[ADDR_WIDTH-1:2])",
        f"                        {wadw}'d0: wb_dat_o <= reg_ctrl;",
        f"                        {wadw}'d1: wb_dat_o <= reg_i_t;",
        f"                        {wadw}'d2: wb_dat_o <= reg_spike_count;",
        *param_reads,
        "                        default: wb_dat_o <= {DATA_WIDTH{1'b0}};",
        "                    endcase",
        "                end",
        "            end",
        "        end",
        "    end",
        "",
        "endmodule",
        "",
    ]
    return "\n".join(lines)
