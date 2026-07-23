# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTOMLLoader from former test_platforms.py

"""Focused suite: TestTOMLLoader from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestTOMLLoader:
    def test_load(self, tmp_path):
        from sc_neurocore.compiler.platforms import load_profiles_from_toml
        from sc_neurocore.compiler.platforms import get_profile

        toml = tmp_path / "custom.toml"
        toml.write_text(
            "[[profile]]\n"
            'name = "test_custom_chip"\n'
            'vendor = "TestVendor"\n'
            'family = "TestFam"\n'
            'platform_class = "custom"\n'
            "data_width = 16\n"
            "fraction = 8\n"
        )
        loaded = load_profiles_from_toml(str(toml))
        assert "test_custom_chip" in loaded
        p = get_profile("test_custom_chip")
        assert p.vendor == "TestVendor"
