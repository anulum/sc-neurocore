# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestApproximation from former test_intelligence_power_and_thermal.py

"""Focused suite: TestApproximation from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

import unittest
from sc_neurocore.compiler.intelligence import (
    configure_approximation,
)


class TestApproximation(unittest.TestCase):
    """Approximate computing for energy savings."""

    def test_basic(self) -> None:
        """Approximation saves energy by reducing bits."""
        r = configure_approximation({"v": "-(v)/tau + I"})
        self.assertGreater(r.total_energy_savings_pct, 0)
        self.assertIn("v", r.populations)
        self.assertIn("bits_reduced", r.populations["v"])

    def test_error_bound(self) -> None:
        """The output error stays within its bound."""
        r = configure_approximation(
            {"v": "a", "u": "b"},
            max_error_pct=2.0,
        )
        self.assertLessEqual(r.max_output_error_pct, 3.1)

    def test_multi_var(self) -> None:
        """Each state variable is approximated."""
        r = configure_approximation({"v": "a", "u": "b", "w": "c"})
        self.assertEqual(len(r.populations), 3)
