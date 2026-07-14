# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live-control parameter-bank generation contracts

"""Static generation and fail-closed validation for live-control RTL."""

from __future__ import annotations

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
                writable=False,
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
    assert "localparam [ADDR_WIDTH-1:0] ADDR_READ_LO    = 32'h124;" in source
    assert "localparam [ADDR_WIDTH-1:0] ADDR_READ_HI    = 32'h128;" in source
    assert "shadow_weights[reg_entry_index] <= staged_word[15:0];" in source
    assert "shadow_kuramoto[reg_entry_index] <= staged_word[31:0];" in source
    assert "reg [DATA_WIDTH-1:0] reg_shadow_bank_select;" in source
    assert "reg [DATA_WIDTH-1:0] reg_shadow_entry_index;" in source
    assert "reg [DATA_WIDTH-1:0] active_read_data_lo;" in source
    assert "reg [DATA_WIDTH-1:0] active_read_data_hi;" in source
    assert "reg_shadow_bank_select <= reg_bank_select;" in source
    assert "reg_shadow_entry_index <= reg_entry_index;" in source
    assert "weights[reg_shadow_entry_index] <= shadow_weights[reg_shadow_entry_index];" in source
    assert "shadow_weights[rollback_entry_index] <= weights[rollback_entry_index];" in source
    assert (
        "wire [DATA_WIDTH-1:0] rollback_bank_select = reg_shadow_loaded ? reg_shadow_bank_select : reg_bank_select;"
        in source
    )
    assert "assign parameter_words[0 +: 16] = weights[0];" in source
    assert "output reg                          apply_pulse" in source
    assert "output reg                          rollback_pulse" in source
    assert "output wire                         shadow_loaded" in source
    assert "output wire [TRAP_WIDTH-1:0]        trap_status_vector" in source
    assert "output wire                         staged_overflow" in source
    assert "output wire                         staged_underflow" in source
    assert "output reg                          checksum_mismatch_pulse" in source
    assert "output reg                          invalid_selection_pulse" in source
    assert "output reg                          read_only_bank_pulse" in source
    assert "output reg                          partial_write_pulse" in source
    assert "FULL_WRITE_STROBE" in source
    assert "wire write_strobe_accepted = (S_AXI_WSTRB == FULL_WRITE_STROBE);" in source
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
    assert "TRAP_READ_ONLY_BANK_VECTOR" in source
    assert "TRAP_PARTIAL_WRITE_VECTOR" in source
    assert "wire kuramoto_writable_for_update = kuramoto_selected_for_update && 1'b0;" in source
    assert (
        "wire bank_update_writable = weights_writable_for_update | kuramoto_writable_for_update;"
        in source
    )
    assert (
        "if (checksum_valid && !staged_update_fault && bank_entry_selection_valid && bank_update_writable) begin"
        in source
    )
    assert "checksum_mismatch_pulse <= 1'b1;" in source
    assert "invalid_selection_pulse <= 1'b1;" in source
    assert "read_only_bank_pulse <= 1'b1;" in source
    assert "partial_write_pulse <= 1'b1;" in source
    assert "reg_trap_vector <= reg_trap_vector | observed_trap_vector;" in source
    assert (
        "ADDR_TRAP_STAT: S_AXI_RDATA <= {{(DATA_WIDTH-TRAP_WIDTH){1'b0}}, reg_trap_vector};"
        in source
    )
    assert (
        "reg_trap_vector <= (reg_trap_vector | observed_trap_vector) & ~S_AXI_WDATA[TRAP_WIDTH-1:0];"
        in source
    )
    assert "ADDR_READ_LO: begin" in source
    assert "ADDR_READ_HI: begin" in source
    assert "S_AXI_RDATA <= active_read_data_lo;" in source
    assert "S_AXI_RDATA <= active_read_data_hi;" in source
    assert "S_AXI_RRESP <= 2'b10;" in source
    assert (
        "reg_trap_vector <= reg_trap_vector | observed_trap_vector | TRAP_INVALID_SELECTION_VECTOR;"
        in source
    )
    assert "trap_clear_pulse <= 1'b1;" in source


def test_live_parameter_bank_rejects_invalid_axi_configuration() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x2000,
        parameter_count=1,
        parameter_names=("w0",),
    )
    base = MMIOUpdateSpec(bus_protocol="axi4_lite", banks=(bank,), control_base_address_bytes=0x100)
    with pytest.raises(ValueError, match="32-bit AXI data bus"):
        generate_live_parameter_bank(base, module_name="m", bus_data_width=64)
    with pytest.raises(ValueError, match="addr_width must be between 8 and 64"):
        generate_live_parameter_bank(base, module_name="m", addr_width=4)
    with pytest.raises(ValueError, match="block_ram_threshold_bits must be positive"):
        generate_live_parameter_bank(base, module_name="m", block_ram_threshold_bits=0)
    wide_trap = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
        trap=TrapSpec(max_flags=40),
    )
    with pytest.raises(ValueError, match="trap flag count must fit"):
        generate_live_parameter_bank(wide_trap, module_name="m")


def test_live_parameter_bank_rejects_invalid_pcie_configuration() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x2000,
        parameter_count=1,
        parameter_names=("w0",),
    )
    spec = MMIOUpdateSpec(bus_protocol="pcie", banks=(bank,), control_base_address_bytes=0x100)
    with pytest.raises(ValueError, match="addr_width must be between 8 and 64"):
        generate_live_parameter_bank(spec, module_name="m", addr_width=4)
    with pytest.raises(ValueError, match="requires 32-bit data"):
        generate_live_parameter_bank(spec, module_name="m", bus_data_width=64)


def test_live_parameter_bank_emits_wide_multiword_bank_read_paths() -> None:
    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        control_base_address_bytes=0x100,
        banks=(
            ParameterBankSpec(
                bank_name="wide48",
                start_address_bytes=0x2000,
                parameter_count=2,
                parameter_names=("a", "b"),
                q_format="Q24.24",
            ),
            ParameterBankSpec(
                bank_name="wide64",
                start_address_bytes=0x3000,
                parameter_count=2,
                parameter_names=("c", "d"),
                q_format="Q32.32",
            ),
        ),
        trap=TrapSpec(max_flags=4),
    )
    source = generate_live_parameter_bank(spec, module_name="sc_wide_params")

    # 48-bit entry: low word plus a zero-padded high-word slice.
    assert "wide48[reg_entry_index][DATA_WIDTH-1:0]" in source
    assert "wide48[reg_entry_index][47:DATA_WIDTH]" in source
    # 64-bit entry: full high-word slice with no padding, and the staged-range
    # extension check is skipped (overflow/underflow tied low).
    assert "wide64[reg_entry_index][63:DATA_WIDTH]" in source
    assert "wire wide64_staged_overflow = 1'b0;" in source
    assert "wire wide64_staged_underflow = 1'b0;" in source


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
    assert "read_only_bank_pulse" in source
    assert "partial_write_pulse" in source
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
    with pytest.raises(ValueError, match="full-word writes"):
        generate_live_parameter_bank(
            MMIOUpdateSpec(
                bus_protocol="axi4_lite",
                banks=(bank,),
                control_base_address_bytes=0x100,
                supports_partial_write=True,
            ),
            module_name="sc_live_params",
        )
