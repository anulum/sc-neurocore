# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for physics modules (heat, wolfram)

import numpy as np
import pytest

from sc_neurocore.physics.heat import StochasticHeatSolver
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph


def _make_uniform_solver(length: float, num_walkers: int, diffusivity: float, seed: int = 0):
    s = StochasticHeatSolver(
        length=length, num_walkers=num_walkers, diffusivity=diffusivity, seed=seed
    )
    s.set_initial_distribution(lambda x: np.ones_like(x))
    return s


class TestStochasticHeatSolver:
    def test_construction(self):
        s = _make_uniform_solver(length=100, num_walkers=500, diffusivity=0.1)
        assert len(s.walkers) == 500
        assert s.length == 100

    def test_step_moves_walkers(self):
        s = _make_uniform_solver(length=100, num_walkers=100, diffusivity=0.1, seed=42)
        pos_before = s.walkers.copy()
        s.step()
        assert not np.array_equal(s.walkers, pos_before)

    def test_walkers_stay_in_bounds(self):
        s = _make_uniform_solver(length=50, num_walkers=1000, diffusivity=0.1, seed=0)
        for _ in range(100):
            s.step()
        assert s.walkers.min() >= 0
        assert s.walkers.max() <= 50

    def test_temperature_profile_shape(self):
        s = _make_uniform_solver(length=20, num_walkers=500, diffusivity=0.1)
        profile = s.get_density(n_bins=20)
        assert profile.shape == (20,)

    def test_temperature_sums_to_one(self):
        s = _make_uniform_solver(length=20, num_walkers=1000, diffusivity=0.1)
        profile = s.get_density(n_bins=20)
        bin_width = s.length / 20
        assert abs(profile.sum() * bin_width - 1.0) < 0.01

    def test_diffusion_spreads(self):
        s = StochasticHeatSolver(length=100, num_walkers=10000, diffusivity=0.1, dt=1e-2, seed=42)
        s.set_initial_delta(50.0)
        p0 = s.get_density(n_bins=100)
        for _ in range(200):
            s.step()
        p1 = s.get_density(n_bins=100)
        # Initial delta concentrates density at one bin; diffusion spreads it.
        assert p1.max() < p0.max()


class TestWolframHypergraph:
    def test_construction(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        assert len(wh.edges) == 2

    def test_evolve_creates_new_nodes(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        wh.evolve(steps=1)
        assert wh.max_node_id > 2

    def test_evolve_creates_new_edges(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        n_edges_before = len(wh.edges)
        wh.evolve(steps=1)
        assert len(wh.edges) >= n_edges_before

    def test_evolve_multiple_steps(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3)], max_node_id=3)
        wh.evolve(steps=5)
        assert len(wh.edges) > 3

    def test_dimension_estimate_small_graph(self):
        wh = WolframHypergraph(edges=[(0, 1)], max_node_id=1)
        d = wh.dimension_estimate()
        assert d == 0.0

    def test_dimension_estimate_after_evolution(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3), (3, 4)], max_node_id=4)
        wh.evolve(steps=3)
        d = wh.dimension_estimate()
        assert d >= 0.0

    def test_no_matching_edges(self):
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
    def test_rejects_invalid_hypergraph_contracts(self, kwargs, match):
        values = {"edges": [(0, 1)], "max_node_id": 1}
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            WolframHypergraph(**values)

    @pytest.mark.parametrize("steps", [-1, 1.5, True])
    def test_evolve_rejects_invalid_steps(self, steps):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        with pytest.raises(ValueError, match="steps"):
            wh.evolve(steps=steps)

    def test_zero_step_evolution_is_identity(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        wh.evolve(steps=0)
        assert wh.edges == [(0, 1), (1, 2)]
        assert wh.max_node_id == 2

    def test_rewrite_rule_preserves_unmatched_edges_and_adds_fresh_node(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (4, 5)], max_node_id=5)

        wh.evolve(steps=1)

        assert (0, 2) in wh.edges
        assert (0, 6) in wh.edges
        assert (1, 6) in wh.edges
        assert (4, 5) in wh.edges
        assert wh.max_node_id == 6

    def test_evolve_skips_non_binary_hyperedges_without_corrupting_rewrite(self):
        wh = WolframHypergraph(edges=[(9, 8, 7), (0, 1), (1, 2)], max_node_id=9)

        wh.evolve(steps=1)

        assert (9, 8, 7) in wh.edges
        assert (0, 2) in wh.edges
        assert wh.max_node_id == 10

    def test_dimension_estimate_rejects_corrupted_edges_before_bfs(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3)], max_node_id=3)
        wh.edges.append((3, 3))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="repeat"):
            wh.dimension_estimate()

    def test_dimension_estimate_returns_zero_for_too_few_nodes(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 0)], max_node_id=2)
        assert wh.dimension_estimate() == 0.0

    def test_dimension_estimate_returns_zero_for_insufficient_growth_depth(self):
        wh = WolframHypergraph(edges=[(0, 1), (2, 3), (4, 5)], max_node_id=5)
        assert wh.dimension_estimate() == 0.0
