# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWolframHypergraph from former test_physics.py

"""Focused suite: TestWolframHypergraph from former test_physics.py."""

from __future__ import annotations

from typing import Any
import pytest
from sc_neurocore.physics.wolfram_hypergraph import (
    WolframHypergraph,
)


class TestWolframHypergraph:
    """The Wolfram-style hypergraph rewriting model."""

    def test_construction(self) -> None:
        """A new hypergraph keeps its initial edges."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        assert len(wh.edges) == 2

    def test_evolve_creates_new_nodes(self) -> None:
        """Evolution allocates fresh node identifiers."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        wh.evolve(steps=1)
        assert wh.max_node_id > 2

    def test_evolve_creates_new_edges(self) -> None:
        """Evolution does not lose edges."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        n_edges_before = len(wh.edges)
        wh.evolve(steps=1)
        assert len(wh.edges) >= n_edges_before

    def test_evolve_multiple_steps(self) -> None:
        """Several evolution steps grow the edge set."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3)], max_node_id=3)
        wh.evolve(steps=5)
        assert len(wh.edges) > 3

    def test_dimension_estimate_small_graph(self) -> None:
        """A small graph has a dimension estimate of zero."""
        wh = WolframHypergraph(edges=[(0, 1)], max_node_id=1)
        d = wh.dimension_estimate()
        assert d == 0.0

    def test_dimension_estimate_after_evolution(self) -> None:
        """The dimension estimate after evolution is not negative."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3), (3, 4)], max_node_id=4)
        wh.evolve(steps=3)
        d = wh.dimension_estimate()
        assert d >= 0.0

    def test_no_matching_edges(self) -> None:
        """Evolution leaves a graph without matching edges unchanged."""
        wh = WolframHypergraph(edges=[(0, 1), (2, 3)], max_node_id=3)
        wh.evolve(steps=1)
        # No chain x->y->z found, edges unchanged
        assert len(wh.edges) == 2

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"edges": "bad"}, "edges"),
            ({"edges": [[]]}, "edges"),
            ({"edges": [(0, True)]}, "integers"),
            ({"edges": [(0, -1)]}, "non-negative"),
            ({"edges": [(0, 0)]}, "repeat"),
            ({"edges": [(0, 3)], "max_node_id": 2}, "largest node"),
            ({"edges": [], "max_node_id": -1}, "max_node_id"),
            ({"edges": [], "max_node_id": 1.5}, "max_node_id"),
        ],
    )
    def test_rejects_invalid_hypergraph_contracts(self, kwargs: dict[str, Any], match: str) -> None:
        """Each invalid construction argument is rejected with its reason."""
        values: dict[str, Any] = {"edges": [(0, 1)], "max_node_id": 1}
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            WolframHypergraph(**values)

    @pytest.mark.parametrize("steps", [-1, 1.5, True])
    def test_evolve_rejects_invalid_steps(self, steps: Any) -> None:
        """A negative, fractional or boolean step count is rejected."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        with pytest.raises(ValueError, match="steps"):
            wh.evolve(steps=steps)

    def test_zero_step_evolution_is_identity(self) -> None:
        """Evolving zero steps leaves the graph unchanged."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        wh.evolve(steps=0)
        assert wh.edges == [(0, 1), (1, 2)]
        assert wh.max_node_id == 2

    def test_rewrite_rule_preserves_unmatched_edges_and_adds_fresh_node(self) -> None:
        """The rewrite keeps unmatched edges and adds one fresh node."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (4, 5)], max_node_id=5)

        wh.evolve(steps=1)

        assert (0, 2) in wh.edges
        assert (0, 6) in wh.edges
        assert (1, 6) in wh.edges
        assert (4, 5) in wh.edges
        assert wh.max_node_id == 6

    def test_evolve_skips_non_binary_hyperedges_without_corrupting_rewrite(self) -> None:
        """Hyperedges that are not binary are skipped without corrupting the rewrite."""
        wh = WolframHypergraph(edges=[(9, 8, 7), (0, 1), (1, 2)], max_node_id=9)

        wh.evolve(steps=1)

        assert (9, 8, 7) in wh.edges
        assert (0, 2) in wh.edges
        assert wh.max_node_id == 10

    def test_dimension_estimate_rejects_corrupted_edges_before_bfs(self) -> None:
        """A corrupted self-loop edge is rejected before the breadth-first search."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3)], max_node_id=3)
        wh.edges.append((3, 3))
        with pytest.raises(ValueError, match="repeat"):
            wh.dimension_estimate()

    def test_dimension_estimate_returns_zero_for_too_few_nodes(self) -> None:
        """Too few nodes give a dimension estimate of zero."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 0)], max_node_id=2)
        assert wh.dimension_estimate() == 0.0

    def test_dimension_estimate_returns_zero_for_insufficient_growth_depth(self) -> None:
        """Too shallow a growth depth gives a dimension estimate of zero."""
        wh = WolframHypergraph(edges=[(0, 1), (2, 3), (4, 5)], max_node_id=5)
        assert wh.dimension_estimate() == 0.0
