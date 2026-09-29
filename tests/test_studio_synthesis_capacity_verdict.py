# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — a synthesised design is judged against its device

"""Synthesis succeeding is not the design fitting its device."""

from __future__ import annotations

from sc_neurocore.studio.synthesis import (
    _DEVICE_CAPACITY,
    capacity_sentence,
    capacity_verdict,
    estimate_resources,
)


def test_the_up5k_has_its_eight_dsp_blocks() -> None:
    # Lattice iCE40 UltraPlus data sheet: the UP5K has 8 SB_MAC16 blocks. The
    # table said 0, which made any multiplier "not fit".
    assert _DEVICE_CAPACITY["ice40"] == {"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 8}


def test_a_design_within_every_capacity_fits() -> None:
    verdict = capacity_verdict(
        {"luts": 1413, "ffs": 85, "brams": 0, "dsps": 0}, _DEVICE_CAPACITY["ice40"]
    )
    assert verdict == {"fits_device": True, "exceeds_capacity": {}}


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
    assert capacity_sentence("ice40", "up5k", verdict["exceeds_capacity"]) == (
        "the design needs 6237 LUTs (the device has 5280), 9 DSP blocks (the device has 8): "
        "it does not fit the ICE40 UP5K"
    )


def test_a_target_without_capacity_data_gets_no_verdict() -> None:
    assert capacity_verdict({"luts": 10**9}, {}) == {}


def test_an_estimate_is_judged_the_same_way() -> None:
    small = estimate_resources(10, "ice40")
    large = estimate_resources(10_000, "ice40")
    assert small["fits_device"] is True
    assert large["fits_device"] is False
    assert large["exceeds_capacity"]["luts"]["needed"] == 10_000 * 2 + 12
