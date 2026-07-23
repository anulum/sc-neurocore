# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProbabilisticPlatforms from former test_platforms.py

"""Focused suite: TestProbabilisticPlatforms from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestProbabilisticPlatforms(unittest.TestCase):
    def test_purdue_pbit(self):
        p = get_profile("purdue_pbit")
        self.assertEqual(p.platform_class, "probabilistic")
        self.assertEqual(p.vendor, "Purdue")

    def test_tohoku_sot_pbit(self):
        p = get_profile("tohoku_sot_pbit")
        self.assertEqual(p.platform_class, "probabilistic")
        self.assertIn("SOT", p.notes)
