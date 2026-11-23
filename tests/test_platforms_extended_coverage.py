# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExtendedCoverage from former test_platforms.py

"""Focused suite: TestExtendedCoverage from former test_platforms.py."""

from __future__ import annotations

import unittest
from sc_neurocore.compiler.platforms import (
    get_profile,
    list_profile_names,
)


class TestExtendedCoverage(unittest.TestCase):
    """Registry growth floors after the extended profile wave."""

    def test_profile_count_ge_183(self) -> None:
        """The registry holds at least 183 profiles."""
        self.assertGreaterEqual(len(list_profile_names()), 183)

    def test_class_count_ge_35(self) -> None:
        """The registry holds at least 35 platform classes."""
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        self.assertGreaterEqual(len(classes), 35)
