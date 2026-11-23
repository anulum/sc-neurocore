# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOptimization from former test_sc_optimizer.py

"""Focused suite: TestOptimization from former test_sc_optimizer.py."""

from __future__ import annotations

import unittest
from sc_neurocore.optimizer.sc_optimizer import (
    HardwareBudget,
    LayerConfig,
    LayerProfile,
    SCOptimizer,
)


class TestOptimization(unittest.TestCase):
    """Greedy optimisation of single and multi-layer networks."""

    def test_single_layer(self) -> None:
        """A one-layer network gets a configuration for its layer."""
        opt = SCOptimizer(HardwareBudget(max_luts=100000, max_power_mw=1000.0))
        network = [LayerProfile(id="L0", mac_count=100)]
        report = opt.optimize(network)
        assert report is not None
        self.assertIsNotNone(report)
        self.assertIn("L0", report.config)
        self.assertIsInstance(report.config["L0"], LayerConfig)

    def test_multi_layer(self) -> None:
        """A ten-layer network gets a configuration for every layer."""
        opt = SCOptimizer(HardwareBudget(max_luts=1000000, max_power_mw=5000.0))
        network = [
            LayerProfile(id=f"L{i}", mac_count=500, is_critical_path=(i == 0)) for i in range(10)
        ]
        report = opt.optimize(network)
        assert report is not None
        self.assertIsNotNone(report)
        self.assertEqual(len(report.config), 10)

    def test_critical_path_gets_priority(self) -> None:
        """A critical-path layer gets at least the accuracy of a normal one."""
        opt = SCOptimizer(HardwareBudget(max_luts=1000000, max_power_mw=50000.0))
        network = [
            LayerProfile(id="critical", mac_count=100, is_critical_path=True),
            LayerProfile(id="normal", mac_count=100, is_critical_path=False),
        ]
        report = opt.optimize(network)
        assert report is not None
        self.assertIsNotNone(report)
        self.assertGreaterEqual(
            report.config["critical"].accuracy_score,
            report.config["normal"].accuracy_score,
        )

    def test_tiny_budget_returns_none(self) -> None:
        """A budget too small for any configuration gives no report."""
        opt = SCOptimizer(HardwareBudget(max_luts=1, max_power_mw=0.001))
        network = [LayerProfile(id="L0", mac_count=10000)]
        result = opt.optimize(network)
        self.assertIsNone(result)

    def test_resource_budget_respected(self) -> None:
        """The report stays within its LUT and power budget."""
        budget = HardwareBudget(max_luts=50000, max_power_mw=500.0)
        opt = SCOptimizer(budget)
        network = [LayerProfile(id=f"L{i}", mac_count=200) for i in range(5)]
        report = opt.optimize(network)
        assert report is not None
        if report:
            self.assertLessEqual(report.total_luts, budget.max_luts)
            self.assertLessEqual(report.total_power_mw, budget.max_power_mw)
