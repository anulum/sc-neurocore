# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFrontierPlatformsW8 from former test_platforms.py

"""Focused suite: TestFrontierPlatformsW8 from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


class TestFrontierPlatformsW8:
    @pytest.mark.parametrize(
        "name",
        [
            "weebit_reram",
            "crossbar_rram",
            "adesto_cbram",
        ],
    )
    def test_rram(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "rram"

    @pytest.mark.parametrize("name", ["tsmc_cim_n7", "samsung_cim_sf3"])
    def test_sram_cim(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "sram_cim"

    @pytest.mark.parametrize("name", ["intel_horse_ridge", "google_cryo_ctrl"])
    def test_cryo_cmos(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "cryo_cmos"

    @pytest.mark.parametrize(
        "name",
        [
            "microsoft_dna_store",
            "asu_dna_perovskite",
        ],
    )
    def test_dna_molecular(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "dna_molecular"

    @pytest.mark.parametrize("name", ["ibm_qnn", "ionq_trapped_ion"])
    def test_quantum_neuro(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "quantum_neuro"

    def test_total_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 155

    def test_total_classes(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 24
