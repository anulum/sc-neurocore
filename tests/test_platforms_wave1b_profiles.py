# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWave1bProfiles from former test_platforms.py

"""Focused suite: TestWave1bProfiles from former test_platforms.py."""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.platforms import (
    get_profile,
)


class TestWave1bProfiles:
    """Verify the 7 additional profiles from §1C/1D."""

    @pytest.mark.parametrize(
        "name",
        [
            "qualcomm_nsp",
            "sambanova",
            "cambricon_mlu",
            "superconducting",
            "cim_sram",
            "analog_ai",
            "event_camera",
        ],
    )
    def test_profile_exists(self, name: str) -> None:
        """Profile is registered and retrievable."""
        p = get_profile(name)
        assert p.name == name
        assert p.data_width > 0

    def test_total_profiles_at_least_84(self) -> None:
        """Registry should now have ≥84 profiles."""
        from sc_neurocore.compiler.platforms import list_profiles

        assert len(list_profiles()) >= 84

    def test_superconducting_is_emerging(self) -> None:
        """Superconducting is in the emerging class."""
        p = get_profile("superconducting")
        assert p.platform_class == "emerging"
        assert p.overflow == "wrap"

    def test_event_camera_matches_dvs(self) -> None:
        """Event camera profile matches DVS sensor specs."""
        p = get_profile("event_camera")
        assert p.vendor == "Prophesee/Sony"
        assert p.data_width == 16
