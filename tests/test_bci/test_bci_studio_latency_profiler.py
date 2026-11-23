# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLatencyProfiler from former test_bci_studio.py

"""Focused suite: TestLatencyProfiler from former test_bci_studio.py."""

from __future__ import annotations

import unittest
from sc_neurocore.bci_studio.bci_studio import (
    LatencyProfiler,
)


class TestLatencyProfiler(unittest.TestCase):
    """Mean, budget and percentiles of the latency profiler."""

    def test_empty_profiler(self) -> None:
        """An empty profiler has a mean of zero."""
        p = LatencyProfiler()
        self.assertEqual(p.mean, 0.0)

    def test_budget_met(self) -> None:
        """Half-millisecond samples meet the latency budget."""
        p = LatencyProfiler()
        for _ in range(100):
            p.record(0.5)
        self.assertTrue(p.budget_met)

    def test_budget_exceeded(self) -> None:
        """Fifteen-millisecond samples exceed the latency budget."""
        p = LatencyProfiler()
        for _ in range(100):
            p.record(15.0)
        self.assertFalse(p.budget_met)

    def test_percentiles(self) -> None:
        """The 99th percentile exceeds the 95th, which exceeds the median."""
        p = LatencyProfiler()
        for i in range(100):
            p.record(float(i))
        self.assertGreater(p.p95, p.p50)
        self.assertGreater(p.p99, p.p95)
