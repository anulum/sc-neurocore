# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTotalProfileCount from former test_platforms.py

"""Focused suite: TestTotalProfileCount from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestTotalProfileCount:
    """Verify total platform coverage."""

    def test_at_least_113_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        names = list_profile_names()
        assert len(names) >= 113, f"Only {len(names)} profiles found"

    def test_10_platform_classes(self):
        from sc_neurocore.compiler.platforms import list_profiles

        classes = {p.platform_class for p in list_profiles()}
        assert len(classes) >= 9

    def test_filter_by_photonic(self):
        from sc_neurocore.compiler.platforms import list_profiles

        photonic = list_profiles(platform_class="photonic")
        assert len(photonic) >= 5

    def test_filter_by_in_memory(self):
        from sc_neurocore.compiler.platforms import list_profiles

        pim = list_profiles(platform_class="in_memory")
        assert len(pim) >= 5
