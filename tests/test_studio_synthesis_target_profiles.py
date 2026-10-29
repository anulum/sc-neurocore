# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public synthesis profile discovery tests

"""Read target definitions without invoking an EDA tool."""

from __future__ import annotations

from sc_neurocore.studio.synthesis import (
    capacity_device_name,
    supported_targets,
    synthesis_target_profiles,
)


def test_profiles_bind_the_exact_registered_commands_and_reference_devices() -> None:
    """Device selectors and routing absence remain visible instead of implied."""
    profiles = synthesis_target_profiles()
    assert tuple(profile.target for profile in profiles) == supported_targets()
    by_name = {profile.target: profile for profile in profiles}
    assert by_name["ice40"].device_selector == "up5k"
    assert by_name["ecp5"].package_selector == "CABGA381"
    assert by_name["ecp5"].routing_tool == "nextpnr-ecp5"
    assert by_name["gowin"].device_selector is None
    assert by_name["xilinx"].device_selector is None
    assert by_name["gowin"].routing_tool is None
    assert by_name["xilinx"].synthesis_command == "synth_xilinx -flatten"
    assert all(
        profile.capacity_device == capacity_device_name(profile.target) for profile in profiles
    )
    by_name["ice40"].capacity["luts"] = 0
    assert synthesis_target_profiles()[0].capacity["luts"] == 5280
