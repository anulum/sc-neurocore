# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AXI4-Lite live parameter-bank renderer

"""Render the validated AXI4-Lite live-control parameter-bank core."""

from __future__ import annotations

import re

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
    TRAP_INVALID_SELECTION,
    TRAP_PARTIAL_WRITE,
    TRAP_READ_ONLY_BANK,
    TRAP_STAGED_OVERFLOW,
    TRAP_STAGED_UNDERFLOW,
)


_SV_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")


def _validate_sv_identifier(name: str, label: str) -> str:
    """Validate and return one SystemVerilog identifier fragment."""
    if not isinstance(name, str) or not _SV_IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"{label} must be a valid SystemVerilog identifier")
    return name


def _ram_style_for_bank(total_bits: int, block_ram_threshold_bits: int) -> str:
    """Return the deterministic RAM style hint for one parameter bank."""
    if total_bits >= block_ram_threshold_bits:
        return "block"
    return "distributed"


def render_axi_live_parameter_bank(
    spec: MMIOUpdateSpec,
    *,
    module_name: str = "sc_live_parameter_bank",
    addr_width: int | None = None,
    bus_data_width: int = 32,
    block_ram_threshold_bits: int = 1024,
) -> str:
    """Render the AXI4-Lite live-parameter bank core.

    Parameters
    ----------
    spec : MMIOUpdateSpec
        Validated live-control register and parameter-bank contract.
    module_name : str
        SystemVerilog module identifier for the generated core.
    addr_width : int or None
        Optional AXI address width override.
    bus_data_width : int
        AXI data width. The maintained core requires 32 bits.
    block_ram_threshold_bits : int
        Minimum bank capacity that receives a block-RAM style hint.

    Returns
    -------
    str
        Complete SystemVerilog source for the AXI4-Lite parameter bank.
    """
    if bus_data_width != 32:
        raise ValueError("live parameter-bank RTL currently requires a 32-bit AXI data bus")
    if spec.supports_partial_write:
        raise ValueError("live parameter-bank RTL requires full-word writes")
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
        "    output reg                          invalid_selection_pulse,",
        "    output reg                          read_only_bank_pulse,",
        "    output reg                          partial_write_pulse,",
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
        f"    localparam [ADDR_WIDTH-1:0] ADDR_READ_LO    = {adw}'h{ctrl['read_data_lo']:X};",
        f"    localparam [ADDR_WIDTH-1:0] ADDR_READ_HI    = {adw}'h{ctrl['read_data_hi']:X};",
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
        f"    localparam [TRAP_WIDTH-1:0] TRAP_INVALID_SELECTION_VECTOR = {trap_width}'h{TRAP_INVALID_SELECTION:X};",
        f"    localparam [TRAP_WIDTH-1:0] TRAP_READ_ONLY_BANK_VECTOR = {trap_width}'h{TRAP_READ_ONLY_BANK:X};",
        f"    localparam [TRAP_WIDTH-1:0] TRAP_PARTIAL_WRITE_VECTOR = {trap_width}'h{TRAP_PARTIAL_WRITE:X};",
        "    localparam [DATA_WIDTH/8-1:0] FULL_WRITE_STROBE = {(DATA_WIDTH/8){1'b1}};",
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
        "    reg [DATA_WIDTH-1:0] reg_shadow_bank_select;",
        "    reg [DATA_WIDTH-1:0] reg_shadow_entry_index;",
        "    reg [TRAP_WIDTH-1:0] reg_trap_vector;",
        "    reg [DATA_WIDTH-1:0] active_read_data_lo;",
        "    reg [DATA_WIDTH-1:0] active_read_data_hi;",
        "    wire [63:0] staged_word = {reg_write_data_hi, reg_write_data_lo};",
        "    wire write_strobe_accepted = (S_AXI_WSTRB == FULL_WRITE_STROBE);",
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
    writable_terms: list[str] = []
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
        writable_terms.append(f"{bank_name}_writable_for_update")
        lines.extend(
            [
                f'    (* ram_style = "{style}" *) reg [{width - 1}:0] {bank_name} [0:{count - 1}];',
                f'    (* ram_style = "{style}" *) reg [{width - 1}:0] shadow_{bank_name} [0:{count - 1}];',
                f"    localparam [{width - 1}:0] RESET_{bank_name.upper()} = {width}'h{reset_word:X};",
                f"    wire {bank_name}_selected_for_update = (reg_bank_select == 32'd{bank_index}) && (reg_entry_index < 32'd{count});",
                f"    wire {bank_name}_writable_for_update = {bank_name}_selected_for_update && 1'b{int(bank.writable)};",
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
            f"    wire bank_entry_selection_valid = {' | '.join(f'{name}_selected_for_update' for name in bank_names)};",
            f"    wire bank_update_writable = {' | '.join(writable_terms)};",
            "    wire [DATA_WIDTH-1:0] rollback_bank_select = reg_shadow_loaded ? reg_shadow_bank_select : reg_bank_select;",
            "    wire [DATA_WIDTH-1:0] rollback_entry_index = reg_shadow_loaded ? reg_shadow_entry_index : reg_entry_index;",
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
            "    always @* begin",
            "        active_read_data_lo = {DATA_WIDTH{1'b0}};",
            "        active_read_data_hi = {DATA_WIDTH{1'b0}};",
            "        case (reg_bank_select)",
        ]
    )
    for bank_index, (bank, bank_name) in enumerate(zip(spec.banks, bank_names)):
        width = bank.entry_width_bits
        count = bank.parameter_count
        lines.extend(
            [
                f"            32'd{bank_index}: begin",
                f"                if (reg_entry_index < 32'd{count}) begin",
            ]
        )
        if width <= bus_data_width:
            pad_width = bus_data_width - width
            if pad_width:
                lines.append(
                    f"                    active_read_data_lo = {{{{{pad_width}{{1'b0}}}}, {bank_name}[reg_entry_index]}};"
                )
            else:
                lines.append(
                    f"                    active_read_data_lo = {bank_name}[reg_entry_index];"
                )
            lines.append("                    active_read_data_hi = {DATA_WIDTH{1'b0}};")
        else:
            high_width = width - bus_data_width
            high_pad_width = bus_data_width - high_width
            lines.append(
                f"                    active_read_data_lo = {bank_name}[reg_entry_index][DATA_WIDTH-1:0];"
            )
            if high_pad_width:
                lines.append(
                    "                    active_read_data_hi = "
                    f"{{{{{high_pad_width}{{1'b0}}}}, {bank_name}[reg_entry_index][{width - 1}:DATA_WIDTH]}};"
                )
            else:
                lines.append(
                    f"                    active_read_data_hi = {bank_name}[reg_entry_index][{width - 1}:DATA_WIDTH];"
                )
        lines.extend(
            [
                "                end",
                "            end",
            ]
        )
    lines.extend(
        [
            "            default: begin end",
            "        endcase",
            "    end",
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
            "            reg_shadow_bank_select <= {DATA_WIDTH{1'b0}};",
            "            reg_shadow_entry_index <= {DATA_WIDTH{1'b0}};",
            "            reg_trap_vector <= {TRAP_WIDTH{1'b0}};",
            "            update_pulse <= 1'b0;",
            "            apply_pulse <= 1'b0;",
            "            rollback_pulse <= 1'b0;",
            "            trap_clear_pulse <= 1'b0;",
            "            checksum_mismatch_pulse <= 1'b0;",
            "            invalid_selection_pulse <= 1'b0;",
            "            read_only_bank_pulse <= 1'b0;",
            "            partial_write_pulse <= 1'b0;",
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
            "            invalid_selection_pulse <= 1'b0;",
            "            read_only_bank_pulse <= 1'b0;",
            "            partial_write_pulse <= 1'b0;",
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
            "                if (!write_strobe_accepted) begin",
            "                    S_AXI_BRESP <= 2'b10;",
            "                    partial_write_pulse <= 1'b1;",
            "                    reg_trap_vector <= reg_trap_vector | observed_trap_vector | TRAP_PARTIAL_WRITE_VECTOR;",
            "                    reg_status <= STATUS_READY | STATUS_TRAP_LATCHED;",
            "                end else begin",
            "                case (S_AXI_AWADDR)",
            "                    ADDR_CONTROL: begin",
            "                        reg_control <= S_AXI_WDATA;",
            "                        if ((S_AXI_WDATA & CTRL_UPDATE_VALID) != {DATA_WIDTH{1'b0}}) begin",
            "                            if (checksum_valid && !staged_update_fault && bank_entry_selection_valid && bank_update_writable) begin",
            "                            update_pulse <= 1'b1;",
            "                            reg_shadow_loaded <= 1'b1;",
            "                            reg_shadow_bank_select <= reg_bank_select;",
            "                            reg_shadow_entry_index <= reg_entry_index;",
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
            "                            else if (!bank_entry_selection_valid) begin",
            "                                invalid_selection_pulse <= 1'b1;",
            "                                reg_trap_vector <= reg_trap_vector | observed_trap_vector | TRAP_INVALID_SELECTION_VECTOR;",
            "                                reg_status <= STATUS_READY | STATUS_TRAP_LATCHED;",
            "                            end",
            "                            else if (!bank_update_writable) begin",
            "                                read_only_bank_pulse <= 1'b1;",
            "                                reg_trap_vector <= reg_trap_vector | observed_trap_vector | TRAP_READ_ONLY_BANK_VECTOR;",
            "                                reg_status <= STATUS_READY | STATUS_TRAP_LATCHED;",
            "                            end",
            "                        end else if ((S_AXI_WDATA & CTRL_COMMIT) != {DATA_WIDTH{1'b0}}) begin",
            "                            if (reg_shadow_loaded) begin",
            "                                apply_pulse <= 1'b1;",
            "                                reg_shadow_loaded <= 1'b0;",
            "                                reg_status <= STATUS_READY | STATUS_UPDATE_ACK | STATUS_APPLIED | trap_status_bit;",
            "                                case (reg_shadow_bank_select)",
        ]
    )

    for bank_index, (bank, bank_name) in enumerate(zip(spec.banks, bank_names)):
        lines.extend(
            [
                f"                                    32'd{bank_index}: begin",
                f"                                        if (reg_shadow_entry_index < 32'd{bank.parameter_count}) begin",
                f"                                            {bank_name}[reg_shadow_entry_index] <= shadow_{bank_name}[reg_shadow_entry_index];",
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
            "                            case (rollback_bank_select)",
        ]
    )

    for bank_index, (bank, bank_name) in enumerate(zip(spec.banks, bank_names)):
        lines.extend(
            [
                f"                                32'd{bank_index}: begin",
                f"                                    if (rollback_entry_index < 32'd{bank.parameter_count}) begin",
                f"                                        shadow_{bank_name}[rollback_entry_index] <= {bank_name}[rollback_entry_index];",
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
            "                        end",
            "                    end",
            "                    ADDR_BANK_SEL: reg_bank_select <= S_AXI_WDATA;",
            "                    ADDR_ENTRY_IDX: reg_entry_index <= S_AXI_WDATA;",
            "                    ADDR_DATA_LO: reg_write_data_lo <= S_AXI_WDATA;",
            "                    ADDR_DATA_HI: reg_write_data_hi <= S_AXI_WDATA;",
            "                    ADDR_CHECKSUM: reg_write_checksum <= S_AXI_WDATA;",
            "                    ADDR_TRAP_CLEAR: begin",
            "                        trap_clear_pulse <= 1'b1;",
            "                        reg_trap_vector <= (reg_trap_vector | observed_trap_vector) & ~S_AXI_WDATA[TRAP_WIDTH-1:0];",
            "                    end",
            "                    default: begin end",
            "                endcase",
            "                end",
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
            "                    ADDR_READ_LO: begin",
            "                        S_AXI_RDATA <= active_read_data_lo;",
            "                        if (!bank_entry_selection_valid) begin",
            "                            S_AXI_RRESP <= 2'b10;",
            "                            invalid_selection_pulse <= 1'b1;",
            "                            reg_trap_vector <= reg_trap_vector | observed_trap_vector | TRAP_INVALID_SELECTION_VECTOR;",
            "                            reg_status <= STATUS_READY | STATUS_TRAP_LATCHED;",
            "                        end",
            "                    end",
            "                    ADDR_READ_HI: begin",
            "                        S_AXI_RDATA <= active_read_data_hi;",
            "                        if (!bank_entry_selection_valid) begin",
            "                            S_AXI_RRESP <= 2'b10;",
            "                            invalid_selection_pulse <= 1'b1;",
            "                            reg_trap_vector <= reg_trap_vector | observed_trap_vector | TRAP_INVALID_SELECTION_VECTOR;",
            "                            reg_status <= STATUS_READY | STATUS_TRAP_LATCHED;",
            "                        end",
            "                    end",
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
