# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermodynamicPlatforms from former test_platforms.py

"""Focused suite: TestThermodynamicPlatforms from former test_platforms.py."""

from __future__ import annotations

import unittest
from sc_neurocore.compiler.platforms import (
    get_profile,
)


class TestThermodynamicPlatforms(unittest.TestCase):
    """The thermodynamic computing platform profiles."""

    def test_extropic_epu(self) -> None:
        """The Extropic profile is an eight-bit thermodynamic platform."""
        p = get_profile("extropic_epu")
        self.assertEqual(p.platform_class, "thermodynamic")
        self.assertEqual(p.vendor, "Extropic")
        self.assertEqual(p.data_width, 8)

    def test_normal_cn101(self) -> None:
        """The Normal profile is a stochastic thermodynamic platform."""
        p = get_profile("normal_cn101")
        self.assertEqual(p.platform_class, "thermodynamic")
        self.assertIn("stochastic", p.notes.lower())
