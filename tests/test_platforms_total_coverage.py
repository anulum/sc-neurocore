# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTotalCoverage from former test_platforms.py

"""Focused suite: TestTotalCoverage from former test_platforms.py."""

from __future__ import annotations

from sc_neurocore.compiler.platforms import (
    get_profile,
    list_profile_names,
)


class TestTotalCoverage:
    """Registry growth floors across every profile wave."""

    def test_min_profiles(self) -> None:
        """The registry holds at least 175 profiles."""
        assert len(list_profile_names()) >= 175

    def test_min_classes(self) -> None:
        """The registry holds at least 31 platform classes."""
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 31
