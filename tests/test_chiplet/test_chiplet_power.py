# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet power-domain RTL contracts

"""Ownership, validation, parse, isolation-delay, and restore-delay tests."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from sc_neurocore.chiplet import PowerDomain, PowerDomainMap, emit_power_gating_sv


def test_power_domain_state_mask_and_map_queries() -> None:
    mapping = PowerDomainMap()
    mapping.add_domain(PowerDomain(0, [0, 1], is_active=True))
    mapping.add_domain(PowerDomain(1, [2, 3], is_active=False))
    assert mapping.domains[0].die_mask == 0b11
    assert not mapping.domains[0].is_gated
    assert mapping.domains[1].is_gated
    assert mapping.active_dies() == [0, 1]
    assert mapping.gated_dies() == [2, 3]
    assert mapping.domain_for_die(2) is mapping.domains[1]
    assert mapping.domain_for_die(99) is None


def test_power_domain_validation_and_unique_ownership() -> None:
    invalid: list[Callable[[], PowerDomain]] = [
        lambda: PowerDomain(-1, [0]),
        lambda: PowerDomain(0, []),
        lambda: PowerDomain(0, [64]),
        lambda: PowerDomain(0, [0, 0]),
        lambda: PowerDomain(0, [0], voltage_mv=0),
    ]
    for constructor in invalid:
        with pytest.raises(ValueError):
            constructor()
    mapping = PowerDomainMap([PowerDomain(0, [0, 1])])
    with pytest.raises(ValueError, match="already assigned"):
        mapping.add_domain(PowerDomain(1, [1, 2]))


def test_power_gating_source_exports_metadata_and_parses(tmp_path: Path) -> None:
    source = emit_power_gating_sv(PowerDomain(3, [0, 2, 5], voltage_mv=725))
    assert "SPDX-License-Identifier" in source
    assert "parameter DOMAIN_ID = 3" in source
    assert "parameter DIE_COUNT = 3" in source
    assert "parameter [63:0] DIE_MASK = 64'h0000000000000025" in source
    assert "parameter VOLTAGE_MV = 725" in source
    source_path = tmp_path / "power_domain.sv"
    source_path.write_text(source)
    subprocess.run(
        ["iverilog", "-g2012", "-tnull", str(source_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _simulate_controller(tmp_path: Path, domain_id: int, body: str) -> None:
    source_path = tmp_path / f"power_domain_{domain_id}.sv"
    testbench_path = tmp_path / f"tb_power_domain_{domain_id}.sv"
    output_path = tmp_path / f"power_domain_{domain_id}.out"
    source_path.write_text(emit_power_gating_sv(PowerDomain(domain_id, [1])))
    testbench_path.write_text(body)
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(output_path), str(source_path), str(testbench_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["vvp", str(output_path)], check=True, capture_output=True, text=True)


def test_power_gating_isolates_before_switch_off(tmp_path: Path) -> None:
    _simulate_controller(
        tmp_path,
        4,
        """
module tb;
    reg clk = 1'b0; reg rst_n = 1'b0; reg enable = 1'b0;
    wire domain_active, isolation_en, power_switch_en;
    integer isolated_cycles = 0; integer i; integer wait_cycles = 0;
    always #1 clk = ~clk;
    sc_chiplet_pwr_domain_4 dut (
        .clk(clk), .rst_n(rst_n), .enable(enable), .domain_active(domain_active),
        .isolation_en(isolation_en), .power_switch_en(power_switch_en));
    initial begin
        repeat (2) @(posedge clk); rst_n = 1'b1; enable = 1'b1;
        while (!domain_active && wait_cycles < 16) begin
            @(posedge clk); wait_cycles = wait_cycles + 1;
        end
        enable = 1'b0;
        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk); if (isolation_en && power_switch_en) isolated_cycles = isolated_cycles + 1;
        end
        if (isolated_cycles !== 4) $fatal(1);
        if (power_switch_en !== 1'b0 || domain_active !== 1'b0 || isolation_en !== 1'b1) $fatal(1);
        $finish;
    end
endmodule
""",
    )


def test_power_gating_waits_before_deisolation(tmp_path: Path) -> None:
    _simulate_controller(
        tmp_path,
        5,
        """
module tb;
    reg clk = 1'b0; reg rst_n = 1'b0; reg enable = 1'b0;
    wire domain_active, isolation_en, power_switch_en;
    integer restore_cycles = 0; integer i;
    always #1 clk = ~clk;
    sc_chiplet_pwr_domain_5 dut (
        .clk(clk), .rst_n(rst_n), .enable(enable), .domain_active(domain_active),
        .isolation_en(isolation_en), .power_switch_en(power_switch_en));
    initial begin
        repeat (2) @(posedge clk); rst_n = 1'b1; enable = 1'b1;
        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk);
            if (isolation_en && power_switch_en && !domain_active) restore_cycles = restore_cycles + 1;
        end
        if (restore_cycles !== 4) $fatal(1);
        if (domain_active !== 1'b1 || isolation_en !== 1'b0 || power_switch_en !== 1'b1) $fatal(1);
        $finish;
    end
endmodule
""",
    )
