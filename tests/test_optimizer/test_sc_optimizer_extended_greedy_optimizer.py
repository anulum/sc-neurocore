# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGreedyOptimizer from former test_sc_optimizer_extended.py

"""Focused suite: TestGreedyOptimizer from former test_sc_optimizer_extended.py."""

from __future__ import annotations

import unittest
from sc_neurocore.optimizer.sc_optimizer import (
    HardwareBudget,
    LayerProfile,
    OptimizerReport,
    SCOptimizer,
)
from sc_optimizer_extended_support import (
    make_network,
)


class TestGreedyOptimizer(unittest.TestCase):
    """The greedy optimizer under hardware budgets."""

    def test_feasible_optimization(self) -> None:
        """A feasible network stays within its LUT and power budget."""
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize(make_network())
        assert report is not None
        self.assertIsNotNone(report)
        self.assertIsInstance(report, OptimizerReport)
        self.assertLessEqual(report.total_luts, budget.max_luts)
        self.assertLessEqual(report.total_power_mw, budget.max_power_mw)

    def test_infeasible_returns_none(self) -> None:
        """An infeasible budget gives no report."""
        budget = HardwareBudget(max_luts=10, max_power_mw=0.001)
        opt = SCOptimizer(budget)
        report = opt.optimize(make_network())
        self.assertIsNone(report)

    def test_critical_path_prioritized(self) -> None:
        """A critical-path layer gets at least the accuracy of a normal one."""
        budget = HardwareBudget(max_luts=10_000_000, max_power_mw=50000.0)
        opt = SCOptimizer(budget)
        net = [
            LayerProfile("crit", 100, is_critical_path=True),
            LayerProfile("norm", 100, is_critical_path=False),
        ]
        report = opt.optimize(net)
        assert report is not None
        self.assertIsNotNone(report)
        crit_acc = report.config["crit"].accuracy_score
        norm_acc = report.config["norm"].accuracy_score
        self.assertGreaterEqual(crit_acc, norm_acc)

    def test_report_summary(self) -> None:
        """The report summary names LUTs and power."""
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
        opt = SCOptimizer(budget)
        report = opt.optimize(make_network(2))
        assert report is not None
        self.assertIsNotNone(report)
        summary = report.summary()
        self.assertIn("LUTs:", summary)
        self.assertIn("Power:", summary)

    def test_latency_constraint(self) -> None:
        """The report keeps within its latency budget."""
        budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0, max_latency_cycles=128)
        opt = SCOptimizer(budget)
        report = opt.optimize(make_network(3))
        assert report is not None
        self.assertIsNotNone(report)
        self.assertLessEqual(report.total_latency_cycles, 128)
