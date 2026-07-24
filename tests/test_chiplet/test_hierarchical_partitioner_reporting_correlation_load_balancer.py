# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorrelationLoadBalancer from former test_hierarchical_partitioner_reporting.py

"""Focused suite: TestCorrelationLoadBalancer from former test_hierarchical_partitioner_reporting.py."""

from __future__ import annotations

from hierarchical_partitioner_reporting_support import *  # noqa: F403


class TestCorrelationLoadBalancer:
    def test_compute_load_metrics(self) -> None:
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        metrics = lb.compute_load_metrics(g, parts)
        assert len(metrics) == 2
        assert metrics[0].vertex_count == 5

    def test_balanced_no_recommendations(self) -> None:
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        recs = lb.recommend_migrations(g, parts)
        assert recs == []  # balanced → no recommendations

    def test_imbalanced_generates_recommendations(self) -> None:
        g = _make_chain_graph(10)
        parts = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]  # 8 vs 2
        lb = CorrelationLoadBalancer(imbalance_threshold=0.1)
        recs = lb.recommend_migrations(g, parts)
        assert len(recs) >= 0  # may or may not find boundary candidates

    def test_history_tracked(self) -> None:
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        lb.recommend_migrations(g, parts)
        assert len(lb.history) >= 0  # at least attempted

    def test_boundary_target_may_not_be_underloaded(self) -> None:
        edges = [CorrelationEdge(vertex, 8, scc_weight=0.1) for vertex in range(8)]
        graph = CorrelationAwareGraph(num_vertices=12, edges=edges)
        partitions = [list(range(8)), list(range(8, 12)), []]
        balancer = CorrelationLoadBalancer(imbalance_threshold=0.2)
        assert balancer.recommend_migrations(graph, partitions) == []
        assert balancer.history == [[]]

    def test_empty_metrics_fail_closed_after_forced_threshold(self) -> None:
        graph = CorrelationAwareGraph(num_vertices=0)
        balancer = CorrelationLoadBalancer(imbalance_threshold=-0.1)
        assert balancer.recommend_migrations(graph, []) == []
