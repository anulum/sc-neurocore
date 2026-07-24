# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFrontierPlatforms from former test_platforms.py

"""Focused suite: TestFrontierPlatforms from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


class TestFrontierPlatforms:
    """Verify 4 new platform classes and 10 new profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "finalspark_neuroplatform",
            "cortical_labs_dishbrain",
        ],
    )
    def test_biological(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class in ("biological", "wetware")

    @pytest.mark.parametrize(
        "name",
        [
            "ibm_ecram",
            "samsung_pcram",
            "stanford_ecram",
        ],
    )
    def test_electrochemical(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "electrochemical"

    @pytest.mark.parametrize(
        "name",
        [
            "cerebras_wse3_ws",
            "tesla_dojo3",
            "tachyum_prodigy",
        ],
    )
    def test_wafer_scale(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "wafer_scale"

    @pytest.mark.parametrize(
        "name",
        [
            "aspinity_aml100",
            "renesas_analog_ai",
        ],
    )
    def test_analog_mixed(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "analog_mixed"

    def test_total_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 144

    def test_total_classes(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 19
