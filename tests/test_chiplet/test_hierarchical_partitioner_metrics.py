# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner metric tests

"""CSR, report-summary, boundary, balance, and communication metrics."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    CSRGraph,
    PartitionReport,
    calculate_comm_volume,
    calculate_imbalance_ratio,
    calculate_mean_boundary_scc,
    calculate_total_boundary_scc,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    make_chain_graph as _make_chain_graph,
)


class TestPartitionReport:
    def test_summary(self) -> None:
        r = PartitionReport(
            num_partitions=4,
            partition_sizes=[25, 25, 25, 25],
            edge_cut=12,
            max_boundary_scc=0.15,
            mean_boundary_scc=0.08,
            total_boundary_scc=0.96,
            imbalance_ratio=0.0,
            comm_volume_bytes=24576,
            comm_messages=12,
            seeds=[0xACE1, 0xBEEF, 0xCAFE, 0xDEAD],
        )
        s = r.summary()
        assert "4" in s
        assert "12" in s
        assert "Imbalance" in s
        assert "Comm" in s


# ── CSRGraph Tests ───────────────────────────────────────────────────


class TestCSRGraph:
    def test_from_edge_list(self) -> None:
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        assert csr.num_vertices == 5
        assert csr.num_edges == 4

    def test_neighbors(self) -> None:
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        n1 = csr.neighbors(1)
        assert 0 in n1
        assert 2 in n1

    def test_degree(self) -> None:
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        assert csr.degree(0) == 1  # endpoint
        assert csr.degree(2) == 2  # middle

    def test_edge_weights(self) -> None:
        g = _make_chain_graph(3, scc=0.5)
        csr = CSRGraph.from_edge_list(3, g.edges)
        scc_0 = csr.edge_scc(0)
        assert len(scc_0) == 1
        assert abs(scc_0[0] - 0.5) < 1e-6

    def test_vertex_weights(self) -> None:
        g = _make_chain_graph(3)
        csr = CSRGraph.from_edge_list(3, g.edges, {0: 2.0, 1: 3.0})
        assert csr.vertex_weights[0] == 2.0
        assert csr.vertex_weights[1] == 3.0
        assert csr.vertex_weights[2] == 1.0  # default

    def test_to_csr(self) -> None:
        g = _make_chain_graph(10)
        csr = g.to_csr()
        assert csr.num_vertices == 10
        assert csr.num_edges == 9


# ── Imbalance Ratio Tests ────────────────────────────────────────────


class TestImbalanceRatio:
    def test_perfect_balance(self) -> None:
        parts = [[0, 1], [2, 3], [4, 5]]
        assert calculate_imbalance_ratio(parts) == 0.0

    def test_imbalanced(self) -> None:
        parts = [[0, 1, 2, 3], [4]]
        ratio = calculate_imbalance_ratio(parts)
        assert ratio > 0.0

    def test_empty(self) -> None:
        assert calculate_imbalance_ratio([]) == 0.0

    def test_single_partition(self) -> None:
        assert calculate_imbalance_ratio([[0, 1, 2]]) == 0.0


# ── Mean/Total Boundary SCC Tests ────────────────────────────────────


class TestBoundarySCCMetrics:
    def test_mean_boundary_scc(self) -> None:
        g = _make_chain_graph(6, scc=0.3)
        parts = [[0, 1, 2], [3, 4, 5]]
        mean_scc = calculate_mean_boundary_scc(g, parts)
        assert mean_scc >= 0.0

    def test_total_boundary_scc(self) -> None:
        g = _make_chain_graph(6, scc=0.4)
        parts = [[0, 1, 2], [3, 4, 5]]
        total_scc = calculate_total_boundary_scc(g, parts)
        assert total_scc >= 0.0

    def test_no_boundary(self) -> None:
        g = _make_chain_graph(4, scc=0.5)
        parts = [list(range(4))]
        assert calculate_mean_boundary_scc(g, parts) == 0.0
        assert calculate_total_boundary_scc(g, parts) == 0.0


# ── Communication Volume Tests ───────────────────────────────────────


class TestCommVolume:
    def test_basic(self) -> None:
        g = _make_chain_graph(6, scc=0.1)
        parts = [[0, 1, 2], [3, 4, 5]]
        cv = calculate_comm_volume(g, parts)
        assert cv["boundary_edges"] >= 1
        assert cv["volume_bytes"] > 0
        assert cv["messages"] == cv["boundary_edges"]

    def test_no_boundary(self) -> None:
        g = _make_chain_graph(4)
        parts = [list(range(4))]
        cv = calculate_comm_volume(g, parts)
        assert cv["boundary_edges"] == 0
        assert cv["volume_bytes"] == 0


# ── Ghost Cell Manager Tests ─────────────────────────────────────────
