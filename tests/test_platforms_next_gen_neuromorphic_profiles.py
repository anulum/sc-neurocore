# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNextGenNeuromorphicProfiles from former test_platforms.py

"""Focused suite: TestNextGenNeuromorphicProfiles from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestNextGenNeuromorphicProfiles:
    """Next-generation neuromorphic platform profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "akida2",
            "spinnaker2",
            "dynapse2",
            "rain_neuromorphic",
            "brainscales2",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "neuromorphic"

    def test_brainscales2_wrap(self):
        """BrainScaleS-2 uses wrap overflow (analog domain)."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("brainscales2")
        assert p.overflow == "wrap"
