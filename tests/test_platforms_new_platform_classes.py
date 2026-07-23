# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNewPlatformClasses from former test_platforms.py

"""Focused suite: TestNewPlatformClasses from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestNewPlatformClasses:
    """Verify all 6 new platform classes and 18 new profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "nist_sfq",
            "northrop_aqfp",
            "josephson_jj",
        ],
    )
    def test_superconducting_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "superconducting"
        assert p.max_freq_mhz >= 5000  # GHz-class

    @pytest.mark.parametrize(
        "name",
        [
            "everspin_stt_mram",
            "samsung_sot_mram",
        ],
    )
    def test_spintronic_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "spintronic"

    @pytest.mark.parametrize("name", ["gf_fefet", "sk_hynix_feram"])
    def test_ferroelectric_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "ferroelectric"

    @pytest.mark.parametrize(
        "name",
        [
            "samsung_cgra",
            "qualcomm_npu_cgra",
            "pact_xtensa",
        ],
    )
    def test_cgra_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "cgra"
        assert p.dsp_block  # CGRAs have PE blocks

    @pytest.mark.parametrize(
        "name",
        [
            "tsmc_soic",
            "intel_foveros",
            "amd_3dv",
        ],
    )
    def test_3d_stacked_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "3d_stacked"

    @pytest.mark.parametrize(
        "name",
        [
            "rp2040",
            "esp32_s3",
            "stm32h7",
            "nrf5340",
            "max78000",
        ],
    )
    def test_edge_mcu_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "edge_mcu"

    @pytest.mark.parametrize(
        "name",
        [
            "sifive_x280",
            "qualcomm_ventana",
            "ainekko_rv",
        ],
    )
    def test_riscv_ai_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "accelerator"

    def test_total_profile_count(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 131

    def test_platform_class_count(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 15
