# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExtendedCoverage from former test_platforms.py

"""Focused suite: TestExtendedCoverage from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestExtendedCoverage(unittest.TestCase):
    def test_profile_count_ge_183(self):
        self.assertGreaterEqual(len(list_profile_names()), 183)

    def test_class_count_ge_35(self):
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        self.assertGreaterEqual(len(classes), 35)
