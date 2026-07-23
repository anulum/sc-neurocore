# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeGraphConv from former test_spike_gnn.py

"""Focused suite: TestSpikeGraphConv from former test_spike_gnn.py."""

from __future__ import annotations

from tests.spike_gnn_support import *  # noqa: F403

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
