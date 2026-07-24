# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSovereignDefenceProfiles from former test_platforms.py

"""Focused suite: TestSovereignDefenceProfiles from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


class TestSovereignDefenceProfiles:
    """Sovereign / defence / aerospace profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "bae_rad750",
            "cobham_ut700",
            "mpfs250t_rt",
            "versal_xqrvc1902",
            "trenz_zynq_space",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "fpga"

    def test_rad750_no_dsp(self):
        """RAD750 has no dedicated DSP blocks."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("bae_rad750")
        assert p.dsp_block == ""
        assert p.dsp_mult_a == 0
