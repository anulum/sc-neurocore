# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPolaritonPlatforms from former test_platforms.py

"""Focused suite: TestPolaritonPlatforms from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestPolaritonPlatforms(unittest.TestCase):
    def test_marvell_polariton(self):
        p = get_profile("marvell_polariton")
        self.assertEqual(p.platform_class, "polariton")
        self.assertEqual(p.vendor, "Marvell")

    def test_stanford_polariton(self):
        p = get_profile("stanford_polariton")
        self.assertEqual(p.platform_class, "polariton")
        self.assertIn("perovskite", p.notes.lower())
