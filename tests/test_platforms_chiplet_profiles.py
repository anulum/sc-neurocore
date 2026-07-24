# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChipletProfiles from former test_platforms.py

"""Focused suite: TestChipletProfiles from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


class TestChipletProfiles:
    """Chiplet / UCIe / heterogeneous integration profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "tenstorrent_blackhole",
            "cerebras_wse3",
            "intel_ponte_vecchio",
            "amd_mi300x",
            "ucie_generic",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "accelerator"

    def test_wse3_frequency(self):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("cerebras_wse3")
        assert p.max_freq_mhz == 1000
