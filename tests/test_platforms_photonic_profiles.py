# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhotonicProfiles from former test_platforms.py

"""Focused suite: TestPhotonicProfiles from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestPhotonicProfiles:
    """Photonic / optical compute platform profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "lightmatter_passage",
            "lightelligence_pace",
            "xanadu_x8",
            "ipronics_smartlight",
            "luminous_computing",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "photonic"
        assert p.data_width > 0
        assert p.fraction < p.data_width

    def test_mzi_dsp_block(self):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("lightmatter_passage")
        assert p.dsp_block == "MZI"
