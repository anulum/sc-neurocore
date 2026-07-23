# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAutomotiveEdgeProfiles from former test_platforms.py

"""Focused suite: TestAutomotiveEdgeProfiles from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestAutomotiveEdgeProfiles:
    """Automotive / edge AI SoC profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "mythic_m1076",
            "mobileye_eyeq6",
            "horizon_j6",
            "ambarella_cv72s",
            "hailo15",
            "syntiant_ndp120",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.data_width > 0
