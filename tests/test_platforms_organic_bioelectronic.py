# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOrganicBioelectronic from former test_platforms.py

"""Focused suite: TestOrganicBioelectronic from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


class TestOrganicBioelectronic:
    @pytest.mark.parametrize("name", ["cambridge_oect", "linkoping_organic"])
    def test_organic(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "organic_bioelectronic"
