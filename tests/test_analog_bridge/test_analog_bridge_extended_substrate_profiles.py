# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSubstrateProfiles from former test_analog_bridge_extended.py

"""Focused suite: TestSubstrateProfiles from former test_analog_bridge_extended.py."""

from __future__ import annotations

from analog_bridge_extended_support import *  # noqa: F403

class TestSubstrateProfiles(unittest.TestCase):
    """Substrate profile contract checks."""

    def test_package_exports_bridge_api(self) -> None:
        """Package entry point exports the documented analog bridge API."""
        expected = [
            "AEREvent",
            "AnalogBridge",
            "AnalogSubstrateProfile",
            "CalibrationRoutine",
            "EventDrivenInterface",
        ]
        self.assertEqual(analog_bridge.__tier__, "research")
        self.assertEqual(analog_bridge.__all__, expected)
        for name in expected:
            self.assertIs(getattr(analog_bridge, name), globals()[name])

    def test_brainscales3(self) -> None:
        """The BrainScaleS-3 profile exposes its DAC and fan-in limits."""
        p = AnalogSubstrateProfile.brainscales3()
        self.assertEqual(p.name, "BrainScaleS-3")
        self.assertEqual(p.dac_resolution, 6)
        self.assertEqual(p.max_fanin, 256)

    def test_dynapse2(self) -> None:
        """The DynapSE-2 profile exposes its DAC resolution."""
        p = AnalogSubstrateProfile.dynapse2()
        self.assertEqual(p.name, "DynapSE-2")
        self.assertEqual(p.dac_resolution, 7)

    def test_profile_constructor(self) -> None:
        """Profile construction configures bridge resolution from the profile."""
        bridge = AnalogBridge(profile=AnalogSubstrateProfile.brainscales3())
        self.assertEqual(bridge.dac_res, 6)
        self.assertEqual(bridge.dac_levels, 64)

    def test_legacy_constructor(self) -> None:
        """Legacy range construction preserves explicit DAC resolution."""
        bridge = AnalogBridge(g_range=(0, 100), v_range=(-80, -40), dac_res=10)
        self.assertEqual(bridge.dac_levels, 1024)
