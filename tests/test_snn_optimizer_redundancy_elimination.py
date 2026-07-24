# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRedundancyElimination from former test_snn_optimizer.py

"""Focused suite: TestRedundancyElimination from former test_snn_optimizer.py."""

from __future__ import annotations

from tests.snn_optimizer_support import *  # noqa: F403


class TestRedundancyElimination:
    def test_merges_identical(self):
        w = np.random.randn(1, 4)
        # 3 identical neurons
        weights = np.vstack([w, w, w + 0.001 * np.random.randn(1, 4)])
        g = SNNGraph(
            layers=[
                LayerNode("h", 4, 3, weights, firing_rates=np.full(3, 0.1)),
                LayerNode("out", 3, 1, np.random.randn(1, 3), firing_rates=np.full(1, 0.1)),
            ]
        )
        result = redundancy_elimination(g, correlation_threshold=0.99)
        assert result.neurons_removed >= 1

    def test_no_redundancy(self):
        g = _make_graph()
        result = redundancy_elimination(g)
        assert result.neurons_removed == 0
