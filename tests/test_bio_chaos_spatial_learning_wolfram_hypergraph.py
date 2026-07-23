# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWolframHypergraph from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestWolframHypergraph from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

from tests.bio_chaos_spatial_learning_support import *  # noqa: F403

class TestWolframHypergraph:
    def test_evolve_changes_edges(self):
        hg = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        hg.evolve(1)
        assert len(hg.edges) != 2

    def test_max_node_id_increments(self):
        hg = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        hg.evolve(1)
        assert hg.max_node_id > 2

    def test_dimension_estimate(self):
        hg = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        # Too few edges for meaningful BFS estimation
        assert hg.dimension_estimate() == 0.0
        hg.evolve(3)
        d = hg.dimension_estimate()
        assert d > 0

    def test_dimension_estimate_few_nodes(self):
        # 3 edges but only 3 nodes → too few for BFS estimation
        hg = WolframHypergraph(edges=[(0, 1), (1, 2), (0, 2)], max_node_id=2)
        assert hg.dimension_estimate() == 0.0

    def test_dimension_estimate_complete_graph(self):
        # 4-node complete graph → BFS reaches all in 1 step → < 2 volumes
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        hg = WolframHypergraph(edges=edges, max_node_id=3)
        assert hg.dimension_estimate() == 0.0

    def test_non_binary_edges_skipped(self):
        hg = WolframHypergraph(edges=[(0, 1, 2), (1, 2), (2, 3)], max_node_id=3)
        hg.evolve(1)
        assert any(len(e) == 3 for e in hg.edges)

    def test_multi_step(self):
        hg = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3), (3, 4)], max_node_id=4)
        hg.evolve(3)
        assert len(hg.edges) > 0
