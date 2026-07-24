# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_platforms.py

"""Module-level tests from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


def test_wave12_hardware_profiles_exist():
    """Verify Wave 12 platforms are loaded properly."""
    wetware1 = get_profile("cortical_labs_dishbrain")
    assert wetware1.platform_class == "wetware"
    assert wetware1.data_width == 8

    molecular = get_profile("biomemory_dna")
    assert molecular.platform_class == "molecular"
    assert molecular.vendor == "Biomemory"
