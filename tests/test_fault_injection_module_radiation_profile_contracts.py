# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRadiationProfileContracts from former test_fault_injection_module.py

"""Focused suite: TestRadiationProfileContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from tests.fault_injection_module_support import *  # noqa: F403

class TestRadiationProfileContracts:
    def test_presets_construct_valid_profiles(self):
        for profile in (
            RadiationProfile.terrestrial(),
            RadiationProfile.leo(),
            RadiationProfile.geo(),
            RadiationProfile.deep_space(),
        ):
            assert profile.name
            assert 0.0 <= profile.ber <= 1.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": ""}, "name"),
            ({"ber": -1e-6}, "ber"),
            ({"ber": 1.01}, "ber"),
            ({"ber": float("nan")}, "ber"),
            ({"description": 1}, "description"),
        ],
    )
    def test_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "name": "LEO",
            "ber": 1e-7,
            "description": "ok",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            RadiationProfile(**values)
