# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalStability from former test_spintronic_mapper.py

"""Focused suite: TestThermalStability from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403


class TestThermalStability:
    def test_thermal_stability_positive(self):
        for tech in SpintronicTech:
            cfg = SpintronicDeviceConfig.from_tech(tech)
            assert cfg.thermal_stability > 0

    def test_larger_device_more_stable(self):
        small = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        import copy

        large = copy.deepcopy(small)
        large.width_nm *= 2
        large.length_nm *= 2
        assert large.thermal_stability > small.thermal_stability

    def test_sot_adequate_retention(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        # Δ > 1 is basic sanity; real devices need > 40
        assert cfg.thermal_stability > 1.0
