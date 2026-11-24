# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeGNNLayer from former test_spike_gnn.py

"""Focused suite: TestSpikeGNNLayer from former test_spike_gnn.py."""

from __future__ import annotations

from sc_neurocore.spike_gnn import (
    SpikeGNNLayer,
)
from tests.spike_gnn_support import (
    _triangle_graph,
)


class TestSpikeGNNLayer:
    """The spiking GNN layer and graph classification."""

    def test_forward(self) -> None:
        """A two-layer GNN gives one output row per node."""
        features, adj = _triangle_graph()
        gnn = SpikeGNNLayer([8, 4, 2], T=4)
        out = gnn.forward(features, adj)
        assert out.shape == (3, 2)
        assert gnn.n_layers == 2

    def test_graph_classify(self) -> None:
        """Graph classification returns a valid class index."""
        features, adj = _triangle_graph()
        gnn = SpikeGNNLayer([8, 4, 3], T=4)
        cls = gnn.graph_classify(features, adj)
        assert 0 <= cls < 3

    def test_single_layer(self) -> None:
        """A single-layer GNN gives the layer's output width."""
        features, adj = _triangle_graph()
        gnn = SpikeGNNLayer([8, 4], T=4)
        out = gnn.forward(features, adj)
        assert out.shape == (3, 4)
