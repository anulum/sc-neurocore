# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetamaterialPlatforms from former test_platforms.py

"""Focused suite: TestMetamaterialPlatforms from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403

class TestMetamaterialPlatforms(unittest.TestCase):
    def test_mit_metamaterial(self):
        p = get_profile("mit_metamaterial")
        self.assertEqual(p.platform_class, "metamaterial")
        self.assertEqual(p.vendor, "MIT")

    def test_penn_acoustic_meta(self):
        p = get_profile("penn_acoustic_meta")
        self.assertEqual(p.platform_class, "metamaterial")
        self.assertIn("acoustic", p.notes.lower())
