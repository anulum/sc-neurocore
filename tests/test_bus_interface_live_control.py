# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for live-control bus interface RTL generation

"""Module-specific tests for live-control parameter-bank RTL emission."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

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
            ),
        ),
        trap=TrapSpec(max_flags=8),
    )

    source = generate_live_parameter_bank(spec, module_name="sc_live_params")

    assert "module sc_live_params" in source
    assert '(* ram_style = "distributed" *) reg [15:0] weights [0:3];' in source
    assert '(* ram_style = "block" *) reg [31:0] kuramoto [0:127];' in source
    assert "localparam [ADDR_WIDTH-1:0] ADDR_CONTROL    = 32'h100;" in source
    assert "localparam [ADDR_WIDTH-1:0] ADDR_BANK_SEL   = 32'h108;" in source
    assert "localparam [ADDR_WIDTH-1:0] ADDR_TRAP_CLEAR = 32'h11C;" in source
    assert "weights[reg_entry_index] <= staged_word[15:0];" in source
    assert "kuramoto[reg_entry_index] <= staged_word[31:0];" in source
    assert "assign parameter_words[0 +: 16] = weights[0];" in source
    assert "trap_clear_pulse <= 1'b1;" in source


def test_live_parameter_bank_rejects_non_axi_and_identifier_injection() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x2000,
        parameter_count=1,
        parameter_names=("w0",),
    )

    with pytest.raises(ValueError, match="axi4_lite"):
        generate_live_parameter_bank(MMIOUpdateSpec(bus_protocol="pcie", banks=(bank,)))

    spec = MMIOUpdateSpec(bus_protocol="axi4_lite", banks=(bank,), control_base_address_bytes=0x100)
    with pytest.raises(ValueError, match="SystemVerilog identifier"):
        generate_live_parameter_bank(spec, module_name="bad;endmodule")


def test_live_parameter_bank_rtl_compiles_with_iverilog(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    if iverilog is None:
        raise AssertionError("iverilog must be available for live-control RTL compile parity")

    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        control_base_address_bytes=0x100,
        banks=(
            ParameterBankSpec(
                bank_name="weights",
                start_address_bytes=0x2000,
                parameter_count=2,
                parameter_names=("w0", "w1"),
                q_format="Q8.8",
            ),
        ),
        trap=TrapSpec(max_flags=4),
    )
    source_path = tmp_path / "sc_live_params.sv"
    sim_path = tmp_path / "sc_live_params.out"
    source_path.write_text(
        generate_live_parameter_bank(spec, module_name="sc_live_params"),
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert compile_result.returncode == 0, compile_result.stderr
