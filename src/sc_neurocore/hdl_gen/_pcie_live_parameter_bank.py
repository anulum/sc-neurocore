# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PCIe live parameter-bank adapter renderer

"""Render the PCIe register-window adapter over the live AXI4-Lite core."""

from __future__ import annotations

from dataclasses import replace

from sc_neurocore.compiler.live_control import MMIOUpdateSpec

from ._live_parameter_bank import (
    _validate_sv_identifier,
    render_axi_live_parameter_bank,
)


def render_pcie_live_parameter_bank(
    spec: MMIOUpdateSpec,
    *,
    module_name: str,
    addr_width: int | None,
    bus_data_width: int,
    block_ram_threshold_bits: int,
) -> str:
    """Render a PCIe-MMIO adapter around the AXI4-Lite parameter-bank core.

    Parameters
    ----------
    spec : MMIOUpdateSpec
        Validated PCIe live-control contract.
    module_name : str
        SystemVerilog module identifier for the generated adapter.
    addr_width : int or None
        Optional MMIO address width override.
    bus_data_width : int
        MMIO data width. The maintained adapter requires 32 bits.
    block_ram_threshold_bits : int
        Minimum parameter-bank capacity that receives a block-RAM hint.

    Returns
    -------
    str
        PCIe adapter source followed by its generated AXI4-Lite core.
    """
    module = _validate_sv_identifier(module_name, "module_name")
    adw = addr_width or spec.address_width_bits
    if adw < 8 or adw > 64:
        raise ValueError("addr_width must be between 8 and 64")
    if bus_data_width != 32:
        raise ValueError("PCIe MMIO live parameter-bank RTL currently requires 32-bit data")
    total_parameter_bits = sum(bank.entry_width_bits * bank.parameter_count for bank in spec.banks)
    trap_width = spec.effective_trap_width
    core_module = f"{module}_axi4_lite_core"
    core_spec = replace(spec, bus_protocol="axi4_lite")
    core_source = render_axi_live_parameter_bank(
        core_spec,
        module_name=core_module,
        addr_width=addr_width,
        bus_data_width=bus_data_width,
        block_ram_threshold_bits=block_ram_threshold_bits,
    )

    wrapper_lines = [
        f"// Auto-generated PCIe MMIO live parameter bank for {module}",
        "// SC-NeuroCore bus interface generator",
        "// PCIe hard IP must expose decoded single-clock MMIO read/write strobes.",
        "// Register semantics match the AXI4-Lite live-control parameter bank exactly.",
        "",
        f"module {module} #(",
        f"    parameter integer ADDR_WIDTH = {adw},",
        f"    parameter integer DATA_WIDTH = {bus_data_width},",
        f"    parameter integer PARAMETER_WORDS_WIDTH = {total_parameter_bits},",
        f"    parameter integer TRAP_WIDTH = {trap_width}",
        ") (",
        "    input  wire                         pcie_clk,",
        "    input  wire                         pcie_resetn,",
        "    input  wire [ADDR_WIDTH-1:0]        pcie_mmio_write_addr,",
        "    input  wire [DATA_WIDTH-1:0]        pcie_mmio_write_data,",
        "    input  wire [DATA_WIDTH/8-1:0]      pcie_mmio_write_strobe,",
        "    input  wire                         pcie_mmio_write_valid,",
        "    output wire                         pcie_mmio_write_ready,",
        "    output wire                         pcie_mmio_write_response_valid,",
        "    output wire                         pcie_mmio_write_error,",
        "    input  wire [ADDR_WIDTH-1:0]        pcie_mmio_read_addr,",
        "    input  wire                         pcie_mmio_read_valid,",
        "    output wire                         pcie_mmio_read_ready,",
        "    output wire [DATA_WIDTH-1:0]        pcie_mmio_read_data,",
        "    output wire                         pcie_mmio_read_data_valid,",
        "    output wire                         pcie_mmio_read_error,",
        "    input  wire [TRAP_WIDTH-1:0]        trap_vector,",
        "    output wire                         trap_latched,",
        "    output wire [TRAP_WIDTH-1:0]        trap_status_vector,",
        "    output wire                         staged_overflow,",
        "    output wire                         staged_underflow,",
        "    output wire                         update_pulse,",
        "    output wire                         apply_pulse,",
        "    output wire                         rollback_pulse,",
        "    output wire                         trap_clear_pulse,",
        "    output wire                         checksum_mismatch_pulse,",
        "    output wire                         invalid_selection_pulse,",
        "    output wire                         read_only_bank_pulse,",
        "    output wire                         partial_write_pulse,",
        "    output wire                         shadow_loaded,",
        "    output wire [PARAMETER_WORDS_WIDTH-1:0] parameter_words",
        ");",
        "",
        "    wire core_awready;",
        "    wire core_wready;",
        "    wire [1:0] core_bresp;",
        "    wire core_bvalid;",
        "    wire core_arready;",
        "    wire [DATA_WIDTH-1:0] core_rdata;",
        "    wire [1:0] core_rresp;",
        "    wire core_rvalid;",
        "",
        "    assign pcie_mmio_write_ready = core_awready & core_wready;",
        "    assign pcie_mmio_write_response_valid = core_bvalid;",
        "    assign pcie_mmio_write_error = |core_bresp;",
        "    assign pcie_mmio_read_ready = core_arready;",
        "    assign pcie_mmio_read_data = core_rdata;",
        "    assign pcie_mmio_read_data_valid = core_rvalid;",
        "    assign pcie_mmio_read_error = |core_rresp;",
        "",
        f"    {core_module} #(",
        "        .ADDR_WIDTH(ADDR_WIDTH),",
        "        .DATA_WIDTH(DATA_WIDTH),",
        "        .PARAMETER_WORDS_WIDTH(PARAMETER_WORDS_WIDTH),",
        "        .TRAP_WIDTH(TRAP_WIDTH)",
        "    ) u_axi4_lite_core (",
        "        .S_AXI_ACLK(pcie_clk),",
        "        .S_AXI_ARESETN(pcie_resetn),",
        "        .S_AXI_AWADDR(pcie_mmio_write_addr),",
        "        .S_AXI_AWVALID(pcie_mmio_write_valid),",
        "        .S_AXI_AWREADY(core_awready),",
        "        .S_AXI_WDATA(pcie_mmio_write_data),",
        "        .S_AXI_WSTRB(pcie_mmio_write_strobe),",
        "        .S_AXI_WVALID(pcie_mmio_write_valid),",
        "        .S_AXI_WREADY(core_wready),",
        "        .S_AXI_BRESP(core_bresp),",
        "        .S_AXI_BVALID(core_bvalid),",
        "        .S_AXI_BREADY(1'b1),",
        "        .S_AXI_ARADDR(pcie_mmio_read_addr),",
        "        .S_AXI_ARVALID(pcie_mmio_read_valid),",
        "        .S_AXI_ARREADY(core_arready),",
        "        .S_AXI_RDATA(core_rdata),",
        "        .S_AXI_RRESP(core_rresp),",
        "        .S_AXI_RVALID(core_rvalid),",
        "        .S_AXI_RREADY(1'b1),",
        "        .trap_vector(trap_vector),",
        "        .trap_latched(trap_latched),",
        "        .trap_status_vector(trap_status_vector),",
        "        .staged_overflow(staged_overflow),",
        "        .staged_underflow(staged_underflow),",
        "        .update_pulse(update_pulse),",
        "        .apply_pulse(apply_pulse),",
        "        .rollback_pulse(rollback_pulse),",
        "        .trap_clear_pulse(trap_clear_pulse),",
        "        .checksum_mismatch_pulse(checksum_mismatch_pulse),",
        "        .invalid_selection_pulse(invalid_selection_pulse),",
        "        .read_only_bank_pulse(read_only_bank_pulse),",
        "        .partial_write_pulse(partial_write_pulse),",
        "        .shadow_loaded(shadow_loaded),",
        "        .parameter_words(parameter_words)",
        "    );",
        "",
        "endmodule",
        "",
    ]
    return "\n".join(wrapper_lines) + "\n" + core_source
