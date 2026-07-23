# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSNNGeneticEvolver from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestSNNGeneticEvolver from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

from tests.bio_chaos_spatial_learning_support import *  # noqa: F403

class TestSNNGeneticEvolver:
    def test_population_size(self):
        evo = SNNGeneticEvolver(lambda: _Individual(), lambda ind: float(np.sum(ind.weights)))
        assert len(evo.population) == 20

    def test_evolve_returns_best(self):
        evo = SNNGeneticEvolver(lambda: _Individual(), lambda ind: float(np.sum(ind.weights)))
        best = evo.evolve(3)
        assert hasattr(best, "weights")

    def test_crossover_mixes(self):
        evo = SNNGeneticEvolver(lambda: _Individual(), lambda ind: 0.0)
        p1, p2 = _Individual(), _Individual()
        p1.weights, p2.weights = np.zeros((4, 4)), np.ones((4, 4))
        child = evo._crossover(p1, p2)
        assert child.weights.shape == (4, 4)

    def test_mutate_within_bounds(self):
        evo = SNNGeneticEvolver(lambda: _Individual(), lambda ind: 0.0)
        ind = _Individual()
        ind.weights = np.full((4, 4), 0.5)
        evo._mutate(ind)
        assert np.all(ind.weights >= 0) and np.all(ind.weights <= 1)

    def test_crossover_no_weights(self):
        evo = SNNGeneticEvolver(lambda: object(), lambda ind: 0.0)
        child = evo._crossover(object(), object())
        assert not hasattr(child, "weights")

    def test_mutate_no_weights(self):
        evo = SNNGeneticEvolver(lambda: object(), lambda ind: 0.0)
        evo._mutate(object())
