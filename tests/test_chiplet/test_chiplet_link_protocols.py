# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet protected-link protocol contracts

"""CDC derivation plus CRC and credit-controller RTL tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sc_neurocore.chiplet import (
    CDCConfig,
    ChipletDie,
    ChipletTopology,
    CreditConfig,
    InterposerLink,
    InterposerTech,
    LinkProtection,
    compute_cdc_configs,
    emit_crc32_sv,
    emit_credit_controller_sv,
)


def test_cdc_configs_follow_clock_relationships() -> None:
    same = compute_cdc_configs(ChipletTopology.ring(3))
    assert all(config.is_mesochronous and config.sync_stages == 2 for config in same.values())
    topology = ChipletTopology(dies=[ChipletDie(0, 200.0), ChipletDie(1, 100.0)])
    topology.add_link(InterposerLink.from_tech(0, 1, InterposerTech.UCIE))
    different = compute_cdc_configs(topology)[(0, 1)]
    assert different.ratio == 2.0
    assert not different.is_mesochronous
    assert different.sync_stages == 3


def test_cdc_missing_die_is_skipped_and_zero_destination_ratio_is_one() -> None:
    topology = ChipletTopology(dies=[ChipletDie(0)])
    topology.add_link(InterposerLink(0, 1))
    assert compute_cdc_configs(topology) == {}
    assert CDCConfig(200.0, 0.0).ratio == 1.0


def test_cdc_validation_fails_closed() -> None:
    with pytest.raises(ValueError):
        CDCConfig(-1.0, 100.0)
    with pytest.raises(ValueError):
        CDCConfig(100.0, -1.0)
    with pytest.raises(ValueError):
        CDCConfig(100.0, 100.0, fifo_depth_log2=0)


@pytest.mark.parametrize(
    ("mode", "overhead"),
    [("none", 0), ("parity", 1), ("crc8", 8), ("crc32", 32), ("secded", 8)],
)
def test_link_protection_modes(mode: str, overhead: int) -> None:
    protection = LinkProtection(mode)
    assert protection.overhead_bits == overhead
    assert 0 < protection.effective_bandwidth_ratio <= 1


def test_unknown_link_protection_fails() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        LinkProtection("unknown")


def test_crc32_emitter_contains_real_polynomial_and_frame_check() -> None:
    source = emit_crc32_sv(64)
    assert "SPDX-License-Identifier" in source
    assert "32'h04C11DB7" in source
    assert "32'hEDB88320" in source
    assert "crc_reg <= crc_next;" in source
    assert "crc_error <= (crc_compare_value != expected_crc);" in source
    assert "Placeholder" not in source
    with pytest.raises(ValueError, match="data_width"):
        emit_crc32_sv(0)


def test_credit_geometry_and_identifier_validation() -> None:
    config = CreditConfig(initial_credits=16, credit_granularity=2)
    assert config.buffer_flits == 32
    assert config.credit_width == 6
    with pytest.raises(ValueError, match="initial_credits"):
        CreditConfig(initial_credits=0)
    with pytest.raises(ValueError, match="credit_granularity"):
        CreditConfig(credit_granularity=0)
    with pytest.raises(ValueError, match="link_name"):
        emit_credit_controller_sv(config, "bad-name")


def test_credit_controller_parses_and_scales_width(tmp_path: Path) -> None:
    source = emit_credit_controller_sv(CreditConfig(initial_credits=1024), "wide_credit")
    assert "parameter CREDIT_W = 11" in source
    assert "next_credits > MAX_FLITS" in source
    source_path = tmp_path / "wide_credit.sv"
    source_path.write_text(source)
    subprocess.run(
        ["iverilog", "-g2012", "-tnull", str(source_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_credit_granularity_controls_accepted_flits(tmp_path: Path) -> None:
    source_path = tmp_path / "credit.sv"
    testbench_path = tmp_path / "tb_credit.sv"
    output_path = tmp_path / "credit.out"
    source_path.write_text(
        emit_credit_controller_sv(CreditConfig(2, credit_granularity=3), "granular")
    )
    testbench_path.write_text(
        """
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg tx_valid = 1'b1;
    reg credit_return = 1'b0;
    wire tx_ready;
    wire [2:0] credits_available;
    integer accepted = 0;
    integer i;
    always #1 clk = ~clk;
    sc_chiplet_credit_granular dut (
        .clk(clk), .rst_n(rst_n), .tx_data(64'd0), .tx_valid(tx_valid),
        .tx_ready(tx_ready), .credit_return(credit_return),
        .credits_available(credits_available)
    );
    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        for (i = 0; i < 7; i = i + 1) begin
            @(negedge clk);
            if (tx_ready) accepted = accepted + 1;
        end
        if (accepted !== 6) $fatal(1);
        $finish;
    end
endmodule
"""
    )
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(output_path), str(source_path), str(testbench_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["vvp", str(output_path)], check=True, capture_output=True, text=True)
