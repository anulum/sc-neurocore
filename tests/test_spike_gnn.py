# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.spike_gnn
from __future__ import annotations

from typing import Any

import numpy as np
from sc_neurocore.spike_gnn import SpikeGNNLayer, SpikeGraphConv


def _triangle_graph() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
    features = np.random.rand(3, 8)
    return features, adj


class TestSpikeGraphConv:
    def test_basic(self) -> None:
        features, adj = _triangle_graph()
        conv = SpikeGraphConv(8, 4)
        out = conv.forward(features, adj, T=4)
        assert out.shape == (3, 4)

    def test_no_edges(self) -> None:
        adj = np.eye(3)
        features = np.random.rand(3, 4)
        conv = SpikeGraphConv(4, 2)
        out = conv.forward(features, adj, T=4)
        assert out.shape == (3, 2)

    def test_large_graph(self) -> None:
        N = 50
        adj = (np.random.rand(N, N) > 0.8).astype(np.float64)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)
        features = np.random.rand(N, 16)
        conv = SpikeGraphConv(16, 8)
        out = conv.forward(features, adj, T=4)
        assert out.shape == (N, 8)


class TestSpikeGNNLayer:
    def test_forward(self) -> None:
        features, adj = _triangle_graph()
        gnn = SpikeGNNLayer([8, 4, 2], T=4)
        out = gnn.forward(features, adj)
        assert out.shape == (3, 2)
        assert gnn.n_layers == 2

    def test_graph_classify(self) -> None:
        features, adj = _triangle_graph()
        gnn = SpikeGNNLayer([8, 4, 3], T=4)
        cls = gnn.graph_classify(features, adj)
        assert 0 <= cls < 3

    def test_single_layer(self) -> None:
        features, adj = _triangle_graph()
        gnn = SpikeGNNLayer([8, 4], T=4)
        out = gnn.forward(features, adj)
        assert out.shape == (3, 4)
