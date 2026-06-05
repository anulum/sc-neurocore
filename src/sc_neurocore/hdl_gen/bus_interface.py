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

from dataclasses import replace
import re
from typing import Literal

from sc_neurocore.compiler.live_control import (
    CONTROL_COMMIT,
    CONTROL_ROLLBACK,
    CONTROL_UPDATE_VALID,
    MMIOUpdateSpec,
    STATUS_APPLIED,
    STATUS_CHECKSUM_VALID,
    STATUS_READY,
    STATUS_ROLLBACK_ACK,
    STATUS_SHADOW_LOADED,
    STATUS_TRAP_LATCHED,
    STATUS_UPDATE_ACK,
    TRAP_CHECKSUM_MISMATCH,
    TRAP_STAGED_OVERFLOW,
    TRAP_STAGED_UNDERFLOW,
)


BusProtocol = Literal["axi_lite", "wishbone"]
_SV_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")


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
            inner_module,
            params,
            data_width,
            addr_width,
            bus_data_width,
        )
    elif bus == "wishbone":
        return _generate_wishbone(
            inner_module,
            params,
            data_width,
            addr_width,
            bus_data_width,
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
    reg_map["CTRL"] = offset
    offset += 4
    reg_map["I_T"] = offset
    offset += 4
    reg_map["SPIKE_COUNT"] = offset
    offset += 4
    for name in params:
        reg_map[name] = offset
        offset += 4
    return reg_map


def generate_live_parameter_bank(
    spec: MMIOUpdateSpec,
    *,
    module_name: str = "sc_live_parameter_bank",
    addr_width: int | None = None,
    bus_data_width: int = 32,
    block_ram_threshold_bits: int = 1024,
) -> str:
    """Generate a live-parameter bank from an MMIO update spec.

    The emitted RTL stores each parameter bank in distributed RAM or BRAM and
    exposes the fixed live-control register map through either AXI4-Lite or a
    PCIe MMIO register-window adapter.  The PCIe path intentionally models the
    endpoint-adapter contract only: upstream PCIe hard IP must decode posted
    writes and reads into the single-clock MMIO strobes exposed here.

    Both protocol paths stage low/high data words, commit updates only after a
    checksum-valid update/apply handshake, and export a flattened
    ``parameter_words`` bus for downstream dense, BFP, or phase-coupling RTL.
    """
    if spec.bus_protocol == "pcie":
        return _generate_pcie_live_parameter_bank(
            spec,
            module_name=module_name,
            addr_width=addr_width,
            bus_data_width=bus_data_width,
            block_ram_threshold_bits=block_ram_threshold_bits,
        )
    if spec.bus_protocol != "axi4_lite":
        raise ValueError("live parameter-bank RTL emission requires axi4_lite or pcie")
    if bus_data_width != 32:
        raise ValueError("live parameter-bank RTL currently requires a 32-bit AXI data bus")
    if spec.trap.max_flags > bus_data_width:
        raise ValueError("trap flag count must fit in one AXI read word for RTL emission")
    module = _validate_sv_identifier(module_name, "module_name")
    adw = addr_width or spec.address_width_bits
    if adw < 8 or adw > 64:
        raise ValueError("addr_width must be between 8 and 64")
    if block_ram_threshold_bits <= 0:
        raise ValueError("block_ram_threshold_bits must be positive")

    bank_names = [_validate_sv_identifier(bank.bank_name, "bank_name") for bank in spec.banks]
    total_parameter_bits = sum(bank.entry_width_bits * bank.parameter_count for bank in spec.banks)
    trap_width = spec.effective_trap_width
    ctrl = spec.control_register_addresses
    status_trap_expr = (
        f"{{{{(DATA_WIDTH-4){{1'b0}}}}, 4'h{STATUS_TRAP_LATCHED:X}}}"
        if STATUS_TRAP_LATCHED < 16
        else f"{bus_data_width}'h{STATUS_TRAP_LATCHED:X}"
    )

    lines = [
        f"// Auto-generated live parameter bank for {module}",
        "// SC-NeuroCore bus interface generator",
        "// Updates are staged and committed through AXI4-Lite control registers.",
        "",
        f"module {module} #(",
        f"    parameter integer ADDR_WIDTH = {adw},",
        f"    parameter integer DATA_WIDTH = {bus_data_width},",
        f"    parameter integer PARAMETER_WORDS_WIDTH = {total_parameter_bits},",
        f"    parameter integer TRAP_WIDTH = {trap_width}",
        ") (",
        "    input  wire                         S_AXI_ACLK,",
        "    input  wire                         S_AXI_ARESETN,",
        "    input  wire [ADDR_WIDTH-1:0]        S_AXI_AWADDR,",
        "    input  wire                         S_AXI_AWVALID,",
        "    output reg                          S_AXI_AWREADY,",
        "    input  wire [DATA_WIDTH-1:0]        S_AXI_WDATA,",
        "    input  wire [DATA_WIDTH/8-1:0]      S_AXI_WSTRB,",
        "    input  wire                         S_AXI_WVALID,",
        "    output reg                          S_AXI_WREADY,",
        "    output reg  [1:0]                   S_AXI_BRESP,",
        "    output reg                          S_AXI_BVALID,",
        "    input  wire                         S_AXI_BREADY,",
        "    input  wire [ADDR_WIDTH-1:0]        S_AXI_ARADDR,",
        "    input  wire                         S_AXI_ARVALID,",
        "    output reg                          S_AXI_ARREADY,",
        "    output reg  [DATA_WIDTH-1:0]        S_AXI_RDATA,",
        "    output reg  [1:0]                   S_AXI_RRESP,",
        "    output reg                          S_AXI_RVALID,",
        "    input  wire                         S_AXI_RREADY,",
        "    input  wire [TRAP_WIDTH-1:0]        trap_vector,",
        "    output wire                         trap_latched,",
        "    output wire [TRAP_WIDTH-1:0]        trap_status_vector,",
        "    output wire                         staged_overflow,",
        "    output wire                         staged_underflow,",
        "    output reg                          update_pulse,",
        "    output reg                          apply_pulse,",
        "    output reg                          rollback_pulse,",
        "    output reg                          trap_clear_pulse,",
        "    output reg                          checksum_mismatch_pulse,",
        "    output wire                         shadow_loaded,",
        "    output wire [PARAMETER_WORDS_WIDTH-1:0] parameter_words",
        ");",
        "",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_CONTROL    = {adw}'h{ctrl['control']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_STATUS     = {adw}'h{ctrl['status']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_BANK_SEL   = {adw}'h{ctrl['bank_select']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_ENTRY_IDX  = {adw}'h{ctrl['entry_index']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_DATA_LO    = {adw}'h{ctrl['write_data_lo']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_DATA_HI    = {adw}'h{ctrl['write_data_hi']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_TRAP_STAT  = {adw}'h{ctrl['trap_status']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_TRAP_CLEAR = {adw}'h{ctrl['trap_clear']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_CHECKSUM   = {adw}'h{ctrl['write_checksum']:X};",
        f"    localparam [DATA_WIDTH-1:0] CTRL_UPDATE_VALID = {bus_data_width}'h{CONTROL_UPDATE_VALID:X};",
        f"    localparam [DATA_WIDTH-1:0] CTRL_COMMIT       = {bus_data_width}'h{CONTROL_COMMIT:X};",
        f"    localparam [DATA_WIDTH-1:0] CTRL_ROLLBACK     = {bus_data_width}'h{CONTROL_ROLLBACK:X};",
        f"    localparam [DATA_WIDTH-1:0] STATUS_READY      = {bus_data_width}'h{STATUS_READY:X};",
        f"    localparam [DATA_WIDTH-1:0] STATUS_TRAP_LATCHED = {bus_data_width}'h{STATUS_TRAP_LATCHED:X};",
        f"    localparam [DATA_WIDTH-1:0] STATUS_UPDATE_ACK = {bus_data_width}'h{STATUS_UPDATE_ACK:X};",
        f"    localparam [DATA_WIDTH-1:0] STATUS_SHADOW_LOADED = {bus_data_width}'h{STATUS_SHADOW_LOADED:X};",
        f"    localparam [DATA_WIDTH-1:0] STATUS_APPLIED    = {bus_data_width}'h{STATUS_APPLIED:X};",
        f"    localparam [DATA_WIDTH-1:0] STATUS_ROLLBACK_ACK = {bus_data_width}'h{STATUS_ROLLBACK_ACK:X};",
        f"    localparam [DATA_WIDTH-1:0] STATUS_CHECKSUM_VALID = {bus_data_width}'h{STATUS_CHECKSUM_VALID:X};",
        f"    localparam [TRAP_WIDTH-1:0] TRAP_STAGED_OVERFLOW_VECTOR = {trap_width}'h{TRAP_STAGED_OVERFLOW:X};",
        f"    localparam [TRAP_WIDTH-1:0] TRAP_STAGED_UNDERFLOW_VECTOR = {trap_width}'h{TRAP_STAGED_UNDERFLOW:X};",
        f"    localparam [TRAP_WIDTH-1:0] TRAP_CHECKSUM_MISMATCH_VECTOR = {trap_width}'h{TRAP_CHECKSUM_MISMATCH:X};",
        "    localparam [31:0] UPDATE_CRC32_POLY_REFLECTED = 32'hEDB88320;",
        "",
        "    reg [DATA_WIDTH-1:0] reg_control;",
        "    reg [DATA_WIDTH-1:0] reg_status;",
        "    reg [DATA_WIDTH-1:0] reg_bank_select;",
        "    reg [DATA_WIDTH-1:0] reg_entry_index;",
        "    reg [DATA_WIDTH-1:0] reg_write_data_lo;",
        "    reg [DATA_WIDTH-1:0] reg_write_data_hi;",
        "    reg [DATA_WIDTH-1:0] reg_write_checksum;",
        "    reg reg_shadow_loaded;",
        "    reg [TRAP_WIDTH-1:0] reg_trap_vector;",
        "    wire [63:0] staged_word = {reg_write_data_hi, reg_write_data_lo};",
        "",
        "    function automatic [31:0] crc32_update_word;",
        "        input [31:0] crc_in;",
        "        input [31:0] data_word;",
        "        integer bit_idx;",
        "        reg [31:0] next_crc;",
        "        begin",
        "            next_crc = crc_in;",
        "            for (bit_idx = 0; bit_idx < 32; bit_idx = bit_idx + 1) begin",
        "                if (next_crc[0] ^ data_word[bit_idx]) begin",
        "                    next_crc = {1'b0, next_crc[31:1]} ^ UPDATE_CRC32_POLY_REFLECTED;",
        "                end else begin",
        "                    next_crc = {1'b0, next_crc[31:1]};",
        "                end",
        "            end",
        "            crc32_update_word = next_crc;",
        "        end",
        "    endfunction",
        "",
        "    function automatic [31:0] live_update_crc32;",
        "        input [31:0] bank_select;",
        "        input [31:0] entry_index;",
        "        input [31:0] data_lo;",
        "        input [31:0] data_hi;",
        "        reg [31:0] crc_state;",
        "        begin",
        "            crc_state = 32'hFFFFFFFF;",
        "            crc_state = crc32_update_word(crc_state, bank_select);",
        "            crc_state = crc32_update_word(crc_state, entry_index);",
        "            crc_state = crc32_update_word(crc_state, data_lo);",
        "            crc_state = crc32_update_word(crc_state, data_hi);",
        "            live_update_crc32 = crc_state ^ 32'hFFFFFFFF;",
        "        end",
        "    endfunction",
        "",
        "    wire [DATA_WIDTH-1:0] observed_checksum = live_update_crc32(reg_bank_select, reg_entry_index, reg_write_data_lo, reg_write_data_hi);",
        "    wire checksum_valid = (reg_write_checksum == observed_checksum);",
        f"    wire [DATA_WIDTH-1:0] trap_status_bit = trap_latched ? {status_trap_expr} : {{DATA_WIDTH{{1'b0}}}};",
        "",
    ]

    flat_offset = 0
    overflow_terms: list[str] = []
    underflow_terms: list[str] = []
    for bank_index, (bank, bank_name) in enumerate(zip(spec.banks, bank_names)):
        style = _ram_style_for_bank(
            bank.parameter_count * bank.entry_width_bits, block_ram_threshold_bits
        )
        width = bank.entry_width_bits
        count = bank.parameter_count
        reset_word = bank.normalise_encoded_word(bank.reset_value)
        overflow_name = f"{bank_name}_staged_overflow"
        underflow_name = f"{bank_name}_staged_underflow"
        overflow_terms.append(overflow_name)
        underflow_terms.append(underflow_name)
        lines.extend(
            [
                f'    (* ram_style = "{style}" *) reg [{width - 1}:0] {bank_name} [0:{count - 1}];',
                f'    (* ram_style = "{style}" *) reg [{width - 1}:0] shadow_{bank_name} [0:{count - 1}];',
                f"    localparam [{width - 1}:0] RESET_{bank_name.upper()} = {width}'h{reset_word:X};",
                f"    wire {bank_name}_selected_for_update = (reg_bank_select == 32'd{bank_index}) && (reg_entry_index < 32'd{count});",
            ]
        )
        if width < 64:
            extension_width = 64 - width
            lines.extend(
                [
                    f"    wire {bank_name}_zero_extension_valid = (staged_word[63:{width}] == {extension_width}'d0);",
                    "    wire %s_sign_extension_valid = (staged_word[63:%d] == {%d{staged_word[%d]}});"
                    % (bank_name, width, extension_width, width - 1),
                    f"    wire {bank_name}_staged_range_valid = {bank_name}_zero_extension_valid || {bank_name}_sign_extension_valid;",
                    f"    wire {overflow_name} = {bank_name}_selected_for_update && !{bank_name}_staged_range_valid && !staged_word[63];",
                    f"    wire {underflow_name} = {bank_name}_selected_for_update && !{bank_name}_staged_range_valid && staged_word[63];",
                ]
            )
        else:
            lines.extend(
                [
                    f"    wire {overflow_name} = 1'b0;",
                    f"    wire {underflow_name} = 1'b0;",
                ]
            )
        for index in range(count):
            lines.append(
                f"    assign parameter_words[{flat_offset} +: {width}] = {bank_name}[{index}];"
            )
            flat_offset += width

    lines.extend(
        [
            f"    wire staged_overflow_fault = {' | '.join(overflow_terms)};",
            f"    wire staged_underflow_fault = {' | '.join(underflow_terms)};",
            "    wire staged_update_fault = staged_overflow_fault | staged_underflow_fault;",
            "    wire [TRAP_WIDTH-1:0] generated_trap_vector =",
            "        (staged_overflow_fault ? TRAP_STAGED_OVERFLOW_VECTOR : {TRAP_WIDTH{1'b0}}) |",
            "        (staged_underflow_fault ? TRAP_STAGED_UNDERFLOW_VECTOR : {TRAP_WIDTH{1'b0}});",
            "    wire [TRAP_WIDTH-1:0] observed_trap_vector = trap_vector | generated_trap_vector;",
            "",
            "    assign trap_latched = |reg_trap_vector;",
            "    assign trap_status_vector = reg_trap_vector;",
            "    assign staged_overflow = staged_overflow_fault;",
            "    assign staged_underflow = staged_underflow_fault;",
            "    assign shadow_loaded = reg_shadow_loaded;",
        ]
    )

    lines.extend(
        [
            "",
            "    integer init_idx;",
            "",
            "    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin",
            "        if (!S_AXI_ARESETN) begin",
            "            S_AXI_AWREADY <= 1'b0;",
            "            S_AXI_WREADY <= 1'b0;",
            "            S_AXI_BRESP <= 2'b00;",
            "            S_AXI_BVALID <= 1'b0;",
            "            S_AXI_ARREADY <= 1'b0;",
            "            S_AXI_RDATA <= {DATA_WIDTH{1'b0}};",
            "            S_AXI_RRESP <= 2'b00;",
            "            S_AXI_RVALID <= 1'b0;",
            "            reg_control <= {DATA_WIDTH{1'b0}};",
            "            reg_status <= STATUS_READY;",
            "            reg_bank_select <= {DATA_WIDTH{1'b0}};",
            "            reg_entry_index <= {DATA_WIDTH{1'b0}};",
            "            reg_write_data_lo <= {DATA_WIDTH{1'b0}};",
            "            reg_write_data_hi <= {DATA_WIDTH{1'b0}};",
            "            reg_write_checksum <= {DATA_WIDTH{1'b0}};",
            "            reg_shadow_loaded <= 1'b0;",
            "            reg_trap_vector <= {TRAP_WIDTH{1'b0}};",
            "            update_pulse <= 1'b0;",
            "            apply_pulse <= 1'b0;",
            "            rollback_pulse <= 1'b0;",
            "            trap_clear_pulse <= 1'b0;",
            "            checksum_mismatch_pulse <= 1'b0;",
        ]
    )

    for bank, bank_name in zip(spec.banks, bank_names):
        lines.extend(
            [
                f"            for (init_idx = 0; init_idx < {bank.parameter_count}; init_idx = init_idx + 1) begin",
                f"                {bank_name}[init_idx] <= RESET_{bank_name.upper()};",
                f"                shadow_{bank_name}[init_idx] <= RESET_{bank_name.upper()};",
                "            end",
            ]
        )

    lines.extend(
        [
            "        end else begin",
            "            S_AXI_AWREADY <= 1'b0;",
            "            S_AXI_WREADY <= 1'b0;",
            "            S_AXI_ARREADY <= 1'b0;",
            "            update_pulse <= 1'b0;",
            "            apply_pulse <= 1'b0;",
            "            rollback_pulse <= 1'b0;",
            "            trap_clear_pulse <= 1'b0;",
            "            checksum_mismatch_pulse <= 1'b0;",
            "            reg_trap_vector <= reg_trap_vector | observed_trap_vector;",
            "            reg_status <= STATUS_READY | trap_status_bit | (reg_shadow_loaded ? STATUS_SHADOW_LOADED : {DATA_WIDTH{1'b0}}) | (checksum_valid ? STATUS_CHECKSUM_VALID : {DATA_WIDTH{1'b0}});",
            "",
            "            if (S_AXI_BREADY && S_AXI_BVALID) begin",
            "                S_AXI_BVALID <= 1'b0;",
            "            end",
            "            if (S_AXI_RREADY && S_AXI_RVALID) begin",
            "                S_AXI_RVALID <= 1'b0;",
            "            end",
            "",
            "            if (S_AXI_AWVALID && S_AXI_WVALID && !S_AXI_BVALID) begin",
            "                S_AXI_AWREADY <= 1'b1;",
            "                S_AXI_WREADY <= 1'b1;",
            "                S_AXI_BVALID <= 1'b1;",
            "                S_AXI_BRESP <= 2'b00;",
            "                case (S_AXI_AWADDR)",
            "                    ADDR_CONTROL: begin",
            "                        reg_control <= S_AXI_WDATA;",
            "                        if ((S_AXI_WDATA & CTRL_UPDATE_VALID) != {DATA_WIDTH{1'b0}}) begin",
            "                            if (checksum_valid && !staged_update_fault) begin",
            "                            update_pulse <= 1'b1;",
            "                            reg_shadow_loaded <= 1'b1;",
            "                            reg_status <= STATUS_READY | STATUS_SHADOW_LOADED | STATUS_CHECKSUM_VALID | trap_status_bit;",
            "                            case (reg_bank_select)",
        ]
    )

    for bank_index, (bank, bank_name) in enumerate(zip(spec.banks, bank_names)):
        lines.extend(
            [
                f"                                32'd{bank_index}: begin",
                f"                                    if (reg_entry_index < 32'd{bank.parameter_count}) begin",
                f"                                        shadow_{bank_name}[reg_entry_index] <= staged_word[{bank.entry_width_bits - 1}:0];",
                "                                    end",
                "                                end",
            ]
        )

    lines.extend(
        [
            "                                default: begin",
            "                                    reg_status <= STATUS_READY | trap_status_bit;",
            "                                end",
            "                            endcase",
            "                            end",
            "                            else if (!checksum_valid) begin",
            "                                checksum_mismatch_pulse <= 1'b1;",
            "                                reg_trap_vector <= reg_trap_vector | observed_trap_vector | TRAP_CHECKSUM_MISMATCH_VECTOR;",
            "                                reg_status <= STATUS_READY | STATUS_TRAP_LATCHED;",
            "                            end",
            "                        end else if ((S_AXI_WDATA & CTRL_COMMIT) != {DATA_WIDTH{1'b0}}) begin",
            "                            if (reg_shadow_loaded) begin",
            "                                apply_pulse <= 1'b1;",
            "                                reg_shadow_loaded <= 1'b0;",
            "                                reg_status <= STATUS_READY | STATUS_UPDATE_ACK | STATUS_APPLIED | trap_status_bit;",
            "                                case (reg_bank_select)",
        ]
    )

    for bank_index, (bank, bank_name) in enumerate(zip(spec.banks, bank_names)):
        lines.extend(
            [
                f"                                    32'd{bank_index}: begin",
                f"                                        if (reg_entry_index < 32'd{bank.parameter_count}) begin",
                f"                                            {bank_name}[reg_entry_index] <= shadow_{bank_name}[reg_entry_index];",
                "                                        end",
                "                                    end",
            ]
        )

    lines.extend(
        [
            "                                    default: begin",
            "                                        reg_status <= STATUS_READY | trap_status_bit;",
            "                                    end",
            "                                endcase",
            "                            end",
            "                        end else if ((S_AXI_WDATA & CTRL_ROLLBACK) != {DATA_WIDTH{1'b0}}) begin",
            "                            rollback_pulse <= 1'b1;",
            "                            reg_shadow_loaded <= 1'b0;",
            "                            reg_status <= STATUS_READY | STATUS_ROLLBACK_ACK | trap_status_bit;",
            "                            case (reg_bank_select)",
        ]
    )

    for bank_index, (bank, bank_name) in enumerate(zip(spec.banks, bank_names)):
        lines.extend(
            [
                f"                                32'd{bank_index}: begin",
                f"                                    if (reg_entry_index < 32'd{bank.parameter_count}) begin",
                f"                                        shadow_{bank_name}[reg_entry_index] <= {bank_name}[reg_entry_index];",
                "                                    end",
                "                                end",
            ]
        )

    lines.extend(
        [
            "                                default: begin",
            "                                    reg_status <= STATUS_READY | trap_status_bit;",
            "                                end",
            "                            endcase",
            "                        end",
            "                        if (S_AXI_WDATA[2]) begin",
            "                            trap_clear_pulse <= 1'b1;",
            "                            reg_trap_vector <= {TRAP_WIDTH{1'b0}};",
            "                        end",
            "                    end",
            "                    ADDR_BANK_SEL: reg_bank_select <= S_AXI_WDATA;",
            "                    ADDR_ENTRY_IDX: reg_entry_index <= S_AXI_WDATA;",
            "                    ADDR_DATA_LO: reg_write_data_lo <= S_AXI_WDATA;",
            "                    ADDR_DATA_HI: reg_write_data_hi <= S_AXI_WDATA;",
            "                    ADDR_CHECKSUM: reg_write_checksum <= S_AXI_WDATA;",
            "                    ADDR_TRAP_CLEAR: begin",
            "                        trap_clear_pulse <= 1'b1;",
            "                        reg_trap_vector <= {TRAP_WIDTH{1'b0}};",
            "                    end",
            "                    default: begin end",
            "                endcase",
            "            end",
            "",
            "            if (S_AXI_ARVALID && !S_AXI_RVALID) begin",
            "                S_AXI_ARREADY <= 1'b1;",
            "                S_AXI_RVALID <= 1'b1;",
            "                S_AXI_RRESP <= 2'b00;",
            "                case (S_AXI_ARADDR)",
            "                    ADDR_CONTROL: S_AXI_RDATA <= reg_control;",
            "                    ADDR_STATUS: S_AXI_RDATA <= reg_status | trap_status_bit;",
            "                    ADDR_BANK_SEL: S_AXI_RDATA <= reg_bank_select;",
            "                    ADDR_ENTRY_IDX: S_AXI_RDATA <= reg_entry_index;",
            "                    ADDR_DATA_LO: S_AXI_RDATA <= reg_write_data_lo;",
            "                    ADDR_DATA_HI: S_AXI_RDATA <= reg_write_data_hi;",
            "                    ADDR_CHECKSUM: S_AXI_RDATA <= observed_checksum;",
            "                    ADDR_TRAP_STAT: S_AXI_RDATA <= {{(DATA_WIDTH-TRAP_WIDTH){1'b0}}, reg_trap_vector};",
            "                    default: S_AXI_RDATA <= {DATA_WIDTH{1'b0}};",
            "                endcase",
            "            end",
            "        end",
            "    end",
            "",
            "endmodule",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_pcie_live_parameter_bank(
    spec: MMIOUpdateSpec,
    *,
    module_name: str,
    addr_width: int | None,
    bus_data_width: int,
    block_ram_threshold_bits: int,
) -> str:
    """Generate a PCIe-MMIO adapter around the AXI4-Lite parameter-bank core."""

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
    core_source = generate_live_parameter_bank(
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
        "        .shadow_loaded(shadow_loaded),",
        "        .parameter_words(parameter_words)",
        "    );",
        "",
        "endmodule",
        "",
    ]
    return "\n".join(wrapper_lines) + "\n" + core_source


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
        "// SC-NeuroCore bus interface generator",
        f"// Bus: AXI4-Lite, Data: {bdw}-bit, Neuron: {ndw}-bit",
        "",
        f"module {wrapper_name} #(",
        f"    parameter ADDR_WIDTH = {adw},",
        f"    parameter DATA_WIDTH = {bdw}",
        ") (",
        "    // AXI4-Lite Slave Interface",
        "    input  wire                    S_AXI_ACLK,",
        "    input  wire                    S_AXI_ARESETN,",
        "    // Write address",
        "    input  wire [ADDR_WIDTH-1:0]   S_AXI_AWADDR,",
        "    input  wire                    S_AXI_AWVALID,",
        "    output reg                     S_AXI_AWREADY,",
        "    // Write data",
        "    input  wire [DATA_WIDTH-1:0]   S_AXI_WDATA,",
        "    input  wire [DATA_WIDTH/8-1:0] S_AXI_WSTRB,",
        "    input  wire                    S_AXI_WVALID,",
        "    output reg                     S_AXI_WREADY,",
        "    // Write response",
        "    output reg  [1:0]              S_AXI_BRESP,",
        "    output reg                     S_AXI_BVALID,",
        "    input  wire                    S_AXI_BREADY,",
        "    // Read address",
        "    input  wire [ADDR_WIDTH-1:0]   S_AXI_ARADDR,",
        "    input  wire                    S_AXI_ARVALID,",
        "    output reg                     S_AXI_ARREADY,",
        "    // Read data",
        "    output reg  [DATA_WIDTH-1:0]   S_AXI_RDATA,",
        "    output reg  [1:0]              S_AXI_RRESP,",
        "    output reg                     S_AXI_RVALID,",
        "    input  wire                    S_AXI_RREADY,",
        "    // Interrupt",
        "    output wire                    irq_spike",
        ");",
        "",
        "    // ── Register file ──────────────────────────────────────",
        f"    reg [{bdw - 1}:0] reg_ctrl;        // 0x00: bit0=enable, bit1=reset",
        f"    reg [{bdw - 1}:0] reg_i_t;         // 0x04: input current",
        f"    reg [{bdw - 1}:0] reg_spike_count; // 0x08: spike counter (read-only)",
    ]

    # Parameter registers
    for i, pname in enumerate(param_list):
        lines.append(
            f"    reg [{bdw - 1}:0] reg_{pname.lower()};{' ' * 4}// 0x{0x0C + i * 4:02X}: {pname}"
        )

    lines.extend(
        [
            "",
            "    // ── Neuron instance ───────────────────────────────────",
            "    wire neuron_clk = S_AXI_ACLK;",
            "    wire neuron_rst = ~S_AXI_ARESETN | reg_ctrl[1];",
            "    wire neuron_en  = reg_ctrl[0];",
            "    wire spike_out;",
            "",
            f"    {inner_module} u_neuron (",
            "        .clk(neuron_clk),",
            "        .rst(neuron_rst),",
            "        .en(neuron_en),",
            f"        .I_t(reg_i_t[{ndw - 1}:0]),",
        ]
    )

    for pname in param_list:
        lines.append(f"        .{pname}(reg_{pname.lower()}[{ndw - 1}:0]),")

    lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma
    lines.extend(
        [
            "    );",
            "",
            "    assign irq_spike = spike_out;",
            "",
            "    // ── Spike counter ─────────────────────────────────────",
            "    always @(posedge S_AXI_ACLK)",
            "        if (!S_AXI_ARESETN || reg_ctrl[1])",
            "            reg_spike_count <= 0;",
            "        else if (spike_out)",
            "            reg_spike_count <= reg_spike_count + 1;",
            "",
            "    // ── AXI4-Lite Write Logic ─────────────────────────────",
            "    reg [ADDR_WIDTH-1:0] aw_addr;",
            "",
            "    always @(posedge S_AXI_ACLK) begin",
            "        if (!S_AXI_ARESETN) begin",
            "            S_AXI_AWREADY <= 1'b0;",
            "            S_AXI_WREADY  <= 1'b0;",
            "            S_AXI_BVALID  <= 1'b0;",
            "            S_AXI_BRESP   <= 2'b00;",
            "            reg_ctrl      <= 0;",
            "            reg_i_t       <= 0;",
        ]
    )

    for pname in param_list:
        lines.append(f"            reg_{pname.lower()} <= 0;")

    lines.extend(
        [
            "        end else begin",
            "            // Address phase",
            "            if (S_AXI_AWVALID && !S_AXI_AWREADY) begin",
            "                S_AXI_AWREADY <= 1'b1;",
            "                aw_addr <= S_AXI_AWADDR;",
            "            end else",
            "                S_AXI_AWREADY <= 1'b0;",
            "",
            "            // Data phase",
            "            if (S_AXI_WVALID && !S_AXI_WREADY) begin",
            "                S_AXI_WREADY <= 1'b1;",
            f"                case (aw_addr[{adw - 1}:2])",
            f"                    {adw - 2}'d0: reg_ctrl <= S_AXI_WDATA;",
            f"                    {adw - 2}'d1: reg_i_t  <= S_AXI_WDATA;",
            "                    // 0x08 = spike_count (read-only)",
        ]
    )

    for i, pname in enumerate(param_list):
        lines.append(f"                    {adw - 2}'d{i + 3}: reg_{pname.lower()} <= S_AXI_WDATA;")

    lines.extend(
        [
            "                    default: ;",
            "                endcase",
            "            end else",
            "                S_AXI_WREADY <= 1'b0;",
            "",
            "            // Response",
            "            if (S_AXI_AWREADY && S_AXI_WREADY && !S_AXI_BVALID) begin",
            "                S_AXI_BVALID <= 1'b1;",
            "                S_AXI_BRESP  <= 2'b00;",
            "            end else if (S_AXI_BREADY && S_AXI_BVALID)",
            "                S_AXI_BVALID <= 1'b0;",
            "        end",
            "    end",
            "",
            "    // ── AXI4-Lite Read Logic ──────────────────────────────",
            "    always @(posedge S_AXI_ACLK) begin",
            "        if (!S_AXI_ARESETN) begin",
            "            S_AXI_ARREADY <= 1'b0;",
            "            S_AXI_RVALID  <= 1'b0;",
            "            S_AXI_RRESP   <= 2'b00;",
            "            S_AXI_RDATA   <= 0;",
            "        end else begin",
            "            if (S_AXI_ARVALID && !S_AXI_ARREADY) begin",
            "                S_AXI_ARREADY <= 1'b1;",
            f"                case (S_AXI_ARADDR[{adw - 1}:2])",
            f"                    {adw - 2}'d0: S_AXI_RDATA <= reg_ctrl;",
            f"                    {adw - 2}'d1: S_AXI_RDATA <= reg_i_t;",
            f"                    {adw - 2}'d2: S_AXI_RDATA <= reg_spike_count;",
        ]
    )

    for i, pname in enumerate(param_list):
        lines.append(f"                    {adw - 2}'d{i + 3}: S_AXI_RDATA <= reg_{pname.lower()};")

    lines.extend(
        [
            "                    default: S_AXI_RDATA <= 0;",
            "                endcase",
            "                S_AXI_RVALID <= 1'b1;",
            "                S_AXI_RRESP  <= 2'b00;",
            "            end else begin",
            "                S_AXI_ARREADY <= 1'b0;",
            "                if (S_AXI_RREADY && S_AXI_RVALID)",
            "                    S_AXI_RVALID <= 1'b0;",
            "            end",
            "        end",
            "    end",
            "",
            "endmodule",
            "",
        ]
    )

    return "\n".join(lines)


def _validate_sv_identifier(name: str, label: str) -> str:
    """Validate a SystemVerilog identifier fragment before emission."""
    if not isinstance(name, str) or not _SV_IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"{label} must be a valid SystemVerilog identifier")
    return name


def _ram_style_for_bank(total_bits: int, block_ram_threshold_bits: int) -> str:
    """Return a deterministic RAM style hint for one parameter bank."""
    if total_bits >= block_ram_threshold_bits:
        return "block"
    return "distributed"


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
        "// SC-NeuroCore bus interface generator",
        f"// Bus: Wishbone B4, Data: {bdw}-bit, Neuron: {ndw}-bit",
        "",
        f"module {wrapper_name} #(",
        f"    parameter ADDR_WIDTH = {adw},",
        f"    parameter DATA_WIDTH = {bdw}",
        ") (",
        "    // Wishbone Slave Interface",
        "    input  wire                    wb_clk_i,",
        "    input  wire                    wb_rst_i,",
        "    input  wire [ADDR_WIDTH-1:0]   wb_adr_i,",
        "    input  wire [DATA_WIDTH-1:0]   wb_dat_i,",
        "    output reg  [DATA_WIDTH-1:0]   wb_dat_o,",
        "    input  wire                    wb_we_i,",
        "    input  wire                    wb_stb_i,",
        "    input  wire                    wb_cyc_i,",
        "    output reg                     wb_ack_o,",
        "    // Interrupt",
        "    output wire                    irq_spike",
        ");",
        "",
        "    // ── Register file ──────────────────────────────────────",
        f"    reg [{bdw - 1}:0] reg_ctrl;",
        f"    reg [{bdw - 1}:0] reg_i_t;",
        f"    reg [{bdw - 1}:0] reg_spike_count;",
    ]

    for i, pname in enumerate(param_list):
        lines.append(f"    reg [{bdw - 1}:0] reg_{pname.lower()};")

    lines.extend(
        [
            "",
            "    // ── Neuron instance ───────────────────────────────────",
            "    wire neuron_rst = wb_rst_i | reg_ctrl[1];",
            "    wire neuron_en  = reg_ctrl[0];",
            "    wire spike_out;",
            "",
            f"    {inner_module} u_neuron (",
            "        .clk(wb_clk_i),",
            "        .rst(neuron_rst),",
            "        .en(neuron_en),",
            f"        .I_t(reg_i_t[{ndw - 1}:0]),",
        ]
    )

    for pname in param_list:
        lines.append(f"        .{pname}(reg_{pname.lower()}[{ndw - 1}:0]),")

    lines[-1] = lines[-1].rstrip(",")
    lines.extend(
        [
            "    );",
            "",
            "    assign irq_spike = spike_out;",
            "",
            "    // ── Spike counter ─────────────────────────────────────",
            "    always @(posedge wb_clk_i)",
            "        if (wb_rst_i || reg_ctrl[1])",
            "            reg_spike_count <= 0;",
            "        else if (spike_out)",
            "            reg_spike_count <= reg_spike_count + 1;",
            "",
            "    // ── Wishbone Bus Logic ────────────────────────────────",
            "    wire wb_access = wb_stb_i && wb_cyc_i;",
            "",
            "    always @(posedge wb_clk_i) begin",
            "        if (wb_rst_i) begin",
            "            wb_ack_o <= 1'b0;",
            "            wb_dat_o <= 0;",
            "            reg_ctrl <= 0;",
            "            reg_i_t  <= 0;",
        ]
    )

    for pname in param_list:
        lines.append(f"            reg_{pname.lower()} <= 0;")

    lines.extend(
        [
            "        end else begin",
            "            wb_ack_o <= 1'b0;",
            "            if (wb_access && !wb_ack_o) begin",
            "                wb_ack_o <= 1'b1;",
            "                if (wb_we_i) begin",
            "                    // Write",
            f"                    case (wb_adr_i[{adw - 1}:2])",
            f"                        {adw - 2}'d0: reg_ctrl <= wb_dat_i;",
            f"                        {adw - 2}'d1: reg_i_t  <= wb_dat_i;",
        ]
    )

    for i, pname in enumerate(param_list):
        lines.append(
            f"                        {adw - 2}'d{i + 3}: reg_{pname.lower()} <= wb_dat_i;"
        )

    lines.extend(
        [
            "                        default: ;",
            "                    endcase",
            "                end else begin",
            "                    // Read",
            f"                    case (wb_adr_i[{adw - 1}:2])",
            f"                        {adw - 2}'d0: wb_dat_o <= reg_ctrl;",
            f"                        {adw - 2}'d1: wb_dat_o <= reg_i_t;",
            f"                        {adw - 2}'d2: wb_dat_o <= reg_spike_count;",
        ]
    )

    for i, pname in enumerate(param_list):
        lines.append(
            f"                        {adw - 2}'d{i + 3}: wb_dat_o <= reg_{pname.lower()};"
        )

    lines.extend(
        [
            "                        default: wb_dat_o <= 0;",
            "                    endcase",
            "                end",
            "            end",
            "        end",
            "    end",
            "",
            "endmodule",
            "",
        ]
    )

    return "\n".join(lines)
