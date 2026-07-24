# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOptimize from former test_snn_optimizer.py

"""Focused suite: TestOptimize from former test_snn_optimizer.py."""

from __future__ import annotations

from tests.snn_optimizer_support import *  # noqa: F403


class TestOptimize:
    def test_full_pipeline(self):
        g = SNNGraph(
            layers=[
                LayerNode(
                    "h1",
                    10,
                    8,
                    np.random.randn(8, 10),
                    firing_rates=np.array([0.1, 0.0, 0.15, 0.0, 0.2, 0.0, 0.12, 0.0]),
                ),
                LayerNode("out", 8, 2, np.random.randn(2, 8), firing_rates=np.full(2, 0.1)),
            ]
        )
        optimized, report = optimize(g)
        assert isinstance(report, OptimizationReport)
        assert report.neurons_before >= report.neurons_after
        s = report.summary()
        assert "SNN Optimizer" in s

    def test_specific_passes(self):
        g = _make_graph()
        _, report = optimize(g, passes=["dead_neuron_elimination"])
        assert len(report.pass_results) == 1

    def test_compression_ratio(self):
        g = _make_graph()
        _, report = optimize(g)
        assert report.compression_ratio >= 1.0

    def test_unknown_pass_ignored(self):
        g = _make_graph()
        _, report = optimize(g, passes=["nonexistent_pass"])
        assert len(report.pass_results) == 0
