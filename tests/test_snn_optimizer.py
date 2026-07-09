# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.snn_optimizer

from __future__ import annotations

import numpy as np

from sc_neurocore.snn_optimizer import (
    SNNGraph,
    LayerNode,
    dead_neuron_elimination,
    layer_fusion,
    redundancy_elimination,
    optimize,
    OptimizationReport,
)


def _make_graph():
    return SNNGraph(
        layers=[
            LayerNode("h1", 10, 8, np.random.randn(8, 10), firing_rates=np.full(8, 0.15)),
            LayerNode("h2", 8, 4, np.random.randn(4, 8), firing_rates=np.full(4, 0.1)),
            LayerNode("out", 4, 2, np.random.randn(2, 4), firing_rates=np.full(2, 0.2)),
        ]
    )


class TestSNNGraph:
    def test_total_params(self):
        g = _make_graph()
        assert g.total_params == 80 + 32 + 8

    def test_total_neurons(self):
        g = _make_graph()
        assert g.total_neurons == 14

    def test_copy(self):
        g = _make_graph()
        c = g.copy()
        c.layers[0].weights[0, 0] = 999
        assert g.layers[0].weights[0, 0] != 999


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


class TestLayerFusion:
    def test_fuses_silent_layer(self):
        g = SNNGraph(
            layers=[
                LayerNode(
                    "h1", 4, 3, np.random.randn(3, 4), firing_rates=np.full(3, 0.001)
                ),  # silent → fusible
                LayerNode("h2", 3, 2, np.random.randn(2, 3), firing_rates=np.full(2, 0.1)),
            ]
        )
        result = layer_fusion(g)
        assert result.layers_fused == 1
        assert len(g.layers) == 1
        assert g.layers[0].weights.shape == (2, 4)

    def test_no_fusion_active_layers(self):
        g = _make_graph()
        result = layer_fusion(g)
        assert result.layers_fused == 0


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
