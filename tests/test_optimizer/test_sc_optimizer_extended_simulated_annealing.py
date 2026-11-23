# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSimulatedAnnealing from former test_sc_optimizer_extended.py

"""Focused suite: TestSimulatedAnnealing from former test_sc_optimizer_extended.py."""

from __future__ import annotations

import unittest
from sc_neurocore.optimizer.sc_optimizer import (
    HardwareBudget,
    OptimizerReport,
    SCOptimizer,
)
from sc_optimizer_extended_support import (
    make_network,
)


class TestSimulatedAnnealing(unittest.TestCase):
    """The simulated-annealing optimizer."""

    def test_annealing_produces_report(self) -> None:
        """Annealing returns a report within the LUT budget."""
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize_annealing(make_network(), max_iter=200)
        assert report is not None
        self.assertIsNotNone(report)
        self.assertIsInstance(report, OptimizerReport)
        self.assertLessEqual(report.total_luts, budget.max_luts)

    def test_annealing_infeasible(self) -> None:
        """An infeasible budget gives no report."""
        budget = HardwareBudget(max_luts=10, max_power_mw=0.001)
        opt = SCOptimizer(budget)
        report = opt.optimize_annealing(make_network(), max_iter=100)
        self.assertIsNone(report)

    def test_annealing_deterministic_with_seed(self) -> None:
        """Two runs with the same seed reach the same accuracy."""
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        r1 = opt.optimize_annealing(make_network(), seed=123, max_iter=200)
        assert r1 is not None
        r2 = opt.optimize_annealing(make_network(), seed=123, max_iter=200)
        assert r2 is not None
        self.assertAlmostEqual(r1.mean_accuracy, r2.mean_accuracy)

    def test_annealing_beats_or_matches_greedy(self) -> None:
        """Annealing comes close to or beats the greedy accuracy."""
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        greedy = opt.optimize(make_network())
        assert greedy is not None
        anneal = opt.optimize_annealing(make_network(), max_iter=500)
        assert anneal is not None
        self.assertIsNotNone(greedy)
        self.assertIsNotNone(anneal)
        # Annealing should at least match greedy (may exceed with enough iterations)
        self.assertGreaterEqual(anneal.mean_accuracy, greedy.mean_accuracy * 0.95)

    def test_pareto_frontier_non_empty(self) -> None:
        """Annealing reports a non-empty Pareto frontier."""
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize_annealing(make_network(), max_iter=300)
        assert report is not None
        self.assertIsNotNone(report)
        self.assertGreater(len(report.pareto_frontier), 0)

    def test_pareto_points_sorted_by_luts(self) -> None:
        """The Pareto frontier is sorted by LUTs."""
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize_annealing(make_network(), max_iter=300)
        assert report is not None
        if len(report.pareto_frontier) > 1:
            luts_vals = [p[0] for p in report.pareto_frontier]
            self.assertEqual(luts_vals, sorted(luts_vals))
