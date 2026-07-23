# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWave1Profiles from former test_platforms.py

"""Focused suite: TestWave1Profiles from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestWave1Profiles:
    """Verify all 12 new hardware profiles are registered."""

    @pytest.mark.parametrize(
        "name",
        [
            "loihi3",
            "northpole",
            "innatera_pulsar",
            "versal_ai_edge",
            "proasic3",
            "trion",
            "titanium",
            "gowin_arora_v",
            "intel_agilex5",
            "nvidia_dla",
            "mediatek_apu",
            "aws_inferentia",
        ],
    )
    def test_profile_exists(self, name):
        """Profile is registered and retrievable."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.name == name
        assert p.data_width > 0
        assert p.fraction >= 0
        assert p.vendor

    def test_total_profiles_at_least_77(self):
        """Total registry should have at least 77 profiles."""
        from sc_neurocore.compiler.platforms import list_profiles

        assert len(list_profiles()) >= 77

    def test_loihi3_is_neuromorphic(self):
        """Loihi 3 should be in the neuromorphic class."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("loihi3")
        assert p.platform_class == "neuromorphic"
        assert p.data_width == 32
        assert p.overflow == "wrap"

    def test_versal_ai_edge_dsp58(self):
        """Versal AI Edge should use DSP58 with 27x24 multiplier."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("versal_ai_edge")
        assert p.dsp_block == "DSP58"
        assert p.dsp_mult_a == 27
        assert p.dsp_mult_b == 24
        assert p.max_freq_mhz == 900
