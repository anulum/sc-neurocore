# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPolaritonPlatforms from former test_platforms.py

"""Focused suite: TestPolaritonPlatforms from former test_platforms.py."""

from __future__ import annotations

import unittest
from sc_neurocore.compiler.platforms import (
    get_profile,
)


class TestPolaritonPlatforms(unittest.TestCase):
    """The polariton platform profiles."""

    def test_marvell_polariton(self) -> None:
        """The Marvell profile is a polariton platform."""
        p = get_profile("marvell_polariton")
        self.assertEqual(p.platform_class, "polariton")
        self.assertEqual(p.vendor, "Marvell")

    def test_stanford_polariton(self) -> None:
        """The Stanford profile is a perovskite polariton platform."""
        p = get_profile("stanford_polariton")
        self.assertEqual(p.platform_class, "polariton")
        self.assertIn("perovskite", p.notes.lower())
