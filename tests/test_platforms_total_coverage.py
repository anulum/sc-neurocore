# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTotalCoverage from former test_platforms.py

"""Focused suite: TestTotalCoverage from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestTotalCoverage:
    def test_min_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 175

    def test_min_classes(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 31
