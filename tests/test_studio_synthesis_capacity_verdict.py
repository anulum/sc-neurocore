# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — a synthesised design is judged against its device

"""Synthesis succeeding is not the design fitting its device."""

from __future__ import annotations

import shutil

import pytest

from sc_neurocore.studio.synthesis import (
    _CAPACITY_DEVICE,
    _DEVICE_CAPACITY,
    _TARGETS,
    capacity_sentence,
    capacity_verdict,
    estimate_resources,
    run_synthesis,
)


def test_the_up5k_has_its_eight_dsp_blocks() -> None:
    # Lattice iCE40 UltraPlus data sheet: the UP5K has 8 SB_MAC16 blocks. The
    # table said 0, which made any multiplier "not fit".
    assert _DEVICE_CAPACITY["ice40"] == {"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 8}


def test_a_design_within_every_capacity_fits() -> None:
    verdict = capacity_verdict(
        {"luts": 1413, "ffs": 85, "brams": 0, "dsps": 0}, _DEVICE_CAPACITY["ice40"]
    )
    assert verdict == {
        "fits_device": True,
        "exceeds_capacity": {},
        "capacity_device": None,
        "uncounted_cells": {},
    }


def test_each_resource_beyond_the_device_is_named_with_both_counts() -> None:
    # The measured 20-neuron network: 6237 LUTs against the UP5K's 5280.
    verdict = capacity_verdict(
        {"luts": 6237, "ffs": 340, "brams": 0, "dsps": 9}, _DEVICE_CAPACITY["ice40"]
    )
    assert verdict["fits_device"] is False
    assert verdict["exceeds_capacity"] == {
        "luts": {"needed": 6237, "available": 5280},
        "dsps": {"needed": 9, "available": 8},
    }
    assert capacity_sentence("ice40", verdict["exceeds_capacity"]) == (
        "the design needs 6237 LUTs (the device has 5280), 9 DSP blocks (the device has 8): "
        "it does not fit the iCE40 UP5K"
    )


def test_a_target_without_capacity_data_gets_no_verdict() -> None:
    assert capacity_verdict({"luts": 10**9}, {}) == {}


def test_an_estimate_is_judged_the_same_way() -> None:
    small = estimate_resources(10, "ice40")
    large = estimate_resources(10_000, "ice40")
    assert small["fits_device"] is True
    assert large["fits_device"] is False
    assert large["exceeds_capacity"]["luts"]["needed"] == 10_000 * 2 + 12


def test_every_target_names_the_device_its_capacity_describes() -> None:
    assert set(_CAPACITY_DEVICE) == set(_TARGETS) == set(_DEVICE_CAPACITY)
    # nextpnr-ecp5 --25k: 24288 LUT4 and TRELLIS_FF (the table said 24576).
    assert _DEVICE_CAPACITY["ecp5"] == {"luts": 24288, "ffs": 24288, "brams": 56, "dsps": 28}
    # Gowin DS102, GW2A-18 (the row said 20736 flip-flops, 41 B-SRAM, no multipliers).
    assert _DEVICE_CAPACITY["gowin"] == {"luts": 20736, "ffs": 15552, "brams": 46, "dsps": 48}
    # XC7A35T: 50 RAMB36, counted as 100 RAMB18 halves.
    assert _DEVICE_CAPACITY["xilinx"] == {"luts": 20800, "ffs": 41600, "brams": 100, "dsps": 90}


def test_cells_of_unknown_cost_leave_the_verdict_open() -> None:
    verdict = capacity_verdict(
        {"luts": 10, "ffs": 10, "brams": 0, "dsps": 0},
        _DEVICE_CAPACITY["gowin"],
        device="Gowin GW2A-18",
        uncounted={"RAM16SDP4": 64},
    )
    assert verdict["fits_device"] is None
    assert verdict["uncounted_cells"] == {"RAM16SDP4": 64}
    assert verdict["capacity_device"] == "Gowin GW2A-18"


def test_counts_that_already_overflow_decide_despite_unknown_cells() -> None:
    verdict = capacity_verdict(
        {"luts": 30_000}, _DEVICE_CAPACITY["gowin"], uncounted={"RAM16SDP4": 1}
    )
    assert verdict["fits_device"] is False


#: A 256x16 memory, a registered 16x16 multiplier and a 24-bit counter: block
#: RAM, a DSP, carry logic and LUTs in one design.
_MIXED_DESIGN = """
module top(input clk, input we, input [7:0] addr, input [15:0] din, input [15:0] a,
           input [15:0] b, output reg [15:0] dout, output reg [31:0] prod, output reg [23:0] cnt);
  reg [15:0] mem [0:255];
  always @(posedge clk) begin
    if (we) mem[addr] <= din;
    dout <= mem[addr];
    prod <= a * b;
    cnt <= cnt + 24'd1;
  end
endmodule
"""


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys is not installed")
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        # nextpnr-ecp5 --25k on the same netlist: 24 LUT4, 56 TRELLIS_FF,
        # 1 DP16KD, 1 MULT18X18D (the 12 carry cells are 2 LUT4s each).
        ("ecp5", {"luts": 24, "ffs": 56, "brams": 1, "dsps": 1}),
        # One B-SRAM, not 64 LUT-RAMs counted as block RAM; wide multiplexers
        # are not LUTs.
        ("gowin", {"brams": 1, "dsps": 0}),
        # The Xilinx target failed on every design ("-json" is not an option
        # of synth_xilinx). 16 RAM256X1S LUT-RAMs are 64 LUTs, not 16 BRAMs.
        ("xilinx", {"brams": 0, "dsps": 1}),
        ("ice40", {"brams": 1, "dsps": 0}),
    ],
)
def test_real_netlists_are_counted_per_family(target: str, expected: dict[str, int]) -> None:
    result = run_synthesis(_MIXED_DESIGN, target)

    assert result["success"] is True, result.get("error")
    assert {key: result["resources"][key] for key in expected} == expected
    assert result["uncounted_cells"] == {}
    assert result["fits_device"] is True
    assert result["capacity_device"] == _CAPACITY_DEVICE[target]


def test_a_yosys_failure_is_told_by_its_error_lines() -> None:
    from sc_neurocore.studio.synthesis import _yosys_failure_message

    # Standard output can end mid-line; the error from standard error follows it.
    log = "help text ... ice40_wrapcarrERROR: Command syntax error: Unknown option.\n> synth\n"
    assert _yosys_failure_message(log) == (
        "Synthesis failed: ERROR: Command syntax error: Unknown option."
    )
    assert _yosys_failure_message("x" * 600).startswith("Synthesis failed. Log:\n")
    assert _yosys_failure_message("x" * 600).endswith("x" * 500)
