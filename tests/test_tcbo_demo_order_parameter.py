# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOrderParameter from former test_tcbo_demo.py

"""Focused suite: TestOrderParameter from former test_tcbo_demo.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.experiments.tcbo_demo_engine import (
    _compute_order_parameter,
)


class TestOrderParameter(unittest.TestCase):
    """The Kuramoto order parameter of the TCBO demo."""

    def test_synchronized(self) -> None:
        """Equal phases give an order parameter above 0.99."""
        theta = np.ones(16) * 1.5
        R = _compute_order_parameter(theta)
        self.assertGreater(R, 0.99)

    def test_uniform(self) -> None:
        """Evenly spread phases give an order parameter below 0.15."""
        theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        R = _compute_order_parameter(theta)
        self.assertLess(R, 0.15)

    def test_range(self) -> None:
        """Random phases give an order parameter inside [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            theta = rng.uniform(0, 2 * np.pi, 16)
            R = _compute_order_parameter(theta)
            self.assertGreaterEqual(R, 0.0)
            self.assertLessEqual(R, 1.0)
