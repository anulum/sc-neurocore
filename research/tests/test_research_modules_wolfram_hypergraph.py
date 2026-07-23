# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWolframHypergraph from former test_research_modules.py

"""Focused suite: TestWolframHypergraph from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestWolframHypergraph:
    def test_construction(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        assert len(wh.edges) == 2

    def test_evolve_creates_new_edges(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        wh.evolve(steps=1)
        # Rule: {x,y},{y,z} -> {x,z},{x,w},{y,w}
        assert len(wh.edges) == 3
        assert wh.max_node_id == 3

    def test_dimension_estimate(self):
        """Exercise real BFS neighbourhood growth estimate (not edge count)."""
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3)], max_node_id=3)
        dim = wh.dimension_estimate()
        assert isinstance(dim, float)
        assert dim >= 0.0
        # Four-node path yields a finite positive slope under log-volume fit.
        assert dim < 10.0
        # Too-small graphs are defined to return 0.0.
        tiny = WolframHypergraph(edges=[(0, 1)], max_node_id=1)
        assert tiny.dimension_estimate() == 0.0

    def test_multi_step_evolution(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3), (3, 4)], max_node_id=4)
        initial_count = len(wh.edges)
        wh.evolve(steps=2)
        # Should grow
        assert len(wh.edges) >= initial_count
