# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRadiationProfiles from former test_fault_injection.py

"""Focused suite: TestRadiationProfiles from former test_fault_injection.py."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403

class TestRadiationProfiles(unittest.TestCase):
    def test_leo(self):
        p = RadiationProfile.leo()
        self.assertEqual(p.name, "LEO")
        self.assertGreater(p.ber, 0)

    def test_geo_higher_than_leo(self):
        self.assertGreater(RadiationProfile.geo().ber, RadiationProfile.leo().ber)

    def test_deep_space_highest(self):
        self.assertGreater(RadiationProfile.deep_space().ber, RadiationProfile.geo().ber)

    def test_terrestrial_lowest(self):
        self.assertLess(RadiationProfile.terrestrial().ber, RadiationProfile.leo().ber)
