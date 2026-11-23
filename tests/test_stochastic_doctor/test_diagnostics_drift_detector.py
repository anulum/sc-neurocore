# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDriftDetector from former test_diagnostics.py

"""Focused suite: TestDriftDetector from former test_diagnostics.py."""

from __future__ import annotations

import unittest
from sc_neurocore.stochastic_doctor.diagnostics import (
    DriftDetector,
)


class TestDriftDetector(unittest.TestCase):
    """Drift detector tests."""

    def test_stable_no_drift(self) -> None:
        """A zero correlation never signals drift."""
        dd = DriftDetector(alpha=0.1, threshold=0.3)
        for _ in range(100):
            self.assertFalse(dd.observe(0.0))

    def test_sustained_correlation_triggers_drift(self) -> None:
        """A sustained high correlation activates drift above threshold."""
        dd = DriftDetector(alpha=0.1, threshold=0.3)
        for _ in range(50):
            dd.observe(0.9)
        self.assertTrue(dd.active)
        self.assertGreater(dd.ema, 0.3)

    def test_reset_clears_state(self) -> None:
        """Resetting clears the average, the active flag and the history."""
        dd = DriftDetector(alpha=0.1, threshold=0.3)
        for _ in range(50):
            dd.observe(0.9)
        dd.reset()
        self.assertEqual(dd.ema, 0.0)
        self.assertFalse(dd.active)
        self.assertEqual(len(dd.history), 0)

    def test_negative_correlation_drift(self) -> None:
        """A sustained negative correlation also activates drift."""
        dd = DriftDetector(alpha=0.1, threshold=0.3)
        for _ in range(50):
            dd.observe(-0.8)
        self.assertTrue(dd.active)

    def test_history_tracking(self) -> None:
        """Every observation is kept in the history."""
        dd = DriftDetector(alpha=0.5, threshold=0.9)
        dd.observe(0.5)
        dd.observe(0.5)
        self.assertEqual(len(dd.history), 2)
