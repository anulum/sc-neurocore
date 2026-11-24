# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTCBOController from former test_tcbo_demo.py

"""Focused suite: TestTCBOController from former test_tcbo_demo.py."""

from __future__ import annotations

import unittest
from sc_neurocore.experiments.tcbo_demo_engine import (
    TCBOController,
)


class TestTCBOController(unittest.TestCase):
    """The TCBO feedback controller of the coupling gain."""

    def test_deficit_increases_kappa(self) -> None:
        """An H1 probability below its target raises the gain."""
        ctrl = TCBOController(tau_h1=0.72)
        kappa = ctrl.step(p_h1=0.3, kappa=1.0, dt=0.01)
        self.assertGreater(kappa, 1.0)

    def test_no_deficit_no_change(self) -> None:
        """An H1 probability above its target leaves the gain unchanged."""
        ctrl = TCBOController(tau_h1=0.72)
        kappa = ctrl.step(p_h1=0.9, kappa=1.0, dt=0.01)
        self.assertAlmostEqual(kappa, 1.0, places=3)

    def test_kappa_clamped(self) -> None:
        """A sustained deficit keeps the gain at or below its maximum."""
        ctrl = TCBOController(tau_h1=0.72, kappa_max=5.0)
        kappa = 4.9
        for _ in range(1000):
            kappa = ctrl.step(p_h1=0.0, kappa=kappa, dt=0.1)
        self.assertLessEqual(kappa, 5.0)

    def test_reset(self) -> None:
        """A reset zeroes the integral term."""
        ctrl = TCBOController()
        ctrl.step(0.1, 1.0, 0.1)
        ctrl.reset()
        self.assertEqual(ctrl._integral, 0.0)
