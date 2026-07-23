# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeadNeuronElimination from former test_snn_optimizer.py

"""Focused suite: TestDeadNeuronElimination from former test_snn_optimizer.py."""

from __future__ import annotations

from tests.snn_optimizer_support import *  # noqa: F403

class TestDeadNeuronElimination:
    def test_removes_dead(self):
        g = SNNGraph(
            layers=[
                LayerNode(
                    "h",
                    4,
                    6,
                    np.random.randn(6, 4),
                    firing_rates=np.array([0.1, 0.0, 0.15, 0.0, 0.2, 0.0]),
                ),
                LayerNode("out", 6, 2, np.random.randn(2, 6), firing_rates=np.full(2, 0.1)),
            ]
        )
        result = dead_neuron_elimination(g)
        assert result.neurons_removed == 3
        assert g.layers[0].n_neurons == 3
        assert g.layers[1].weights.shape[1] == 3

    def test_no_dead(self):
        g = _make_graph()
        result = dead_neuron_elimination(g)
        assert result.neurons_removed == 0

    def test_no_firing_rates(self):
        g = SNNGraph(layers=[LayerNode("h", 4, 4, np.random.randn(4, 4))])
        result = dead_neuron_elimination(g)
        assert result.neurons_removed == 0
