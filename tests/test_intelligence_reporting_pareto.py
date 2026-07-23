# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPareto from former test_intelligence_reporting.py

"""Focused suite: TestPareto from former test_intelligence_reporting.py."""

from __future__ import annotations

from tests.intelligence_reporting_support import *  # noqa: F403

class TestPareto(unittest.TestCase):
    def test_non_empty(self):
        pts = explore_pareto({"v": "-(v)/tau + I"})
        self.assertGreater(len(pts), 0)

    def test_non_dominated(self):
        pts = explore_pareto({"v": "a", "u": "b"})
        for i, p in enumerate(pts):
            for j, q in enumerate(pts):
                if i != j:
                    self.assertFalse(
                        q.power_mw <= p.power_mw
                        and q.area_luts <= p.area_luts
                        and q.latency_ns <= p.latency_ns
                        and (
                            q.power_mw < p.power_mw
                            or q.area_luts < p.area_luts
                            or q.latency_ns < p.latency_ns
                        ),
                        f"Point {i} dominated by {j}",
                    )

    def test_sorted_by_power(self):
        pts = explore_pareto({"v": "a"})
        powers = [p.power_mw for p in pts]
        self.assertEqual(powers, sorted(powers))
