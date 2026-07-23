# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSNNGeneticEvolver from former test_research_modules.py

"""Focused suite: TestSNNGeneticEvolver from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestSNNGeneticEvolver:
    def _make_layer(self):
        class MockLayer:
            def __init__(self):
                self.weights = np.random.rand(3, 3)

        return MockLayer()

    def test_construction(self):
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        assert len(evolver.population) == 20

    def test_evolve_returns_best(self, capsys):
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        best = evolver.evolve(generations=3)
        assert hasattr(best, "weights")

    def test_crossover(self):
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        p1 = self._make_layer()
        p2 = self._make_layer()
        child = evolver._crossover(p1, p2)
        assert child.weights.shape == (3, 3)

    def test_mutate(self):
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: 0.0,
        )
        evolver.mutation_rate = 1.0  # mutate every weight
        layer = self._make_layer()
        original = layer.weights.copy()
        evolver._mutate(layer)
        # With 100% mutation rate, weights should change
        assert not np.array_equal(layer.weights, original)
        # Weights should stay in [0, 1]
        assert np.all(layer.weights >= 0) and np.all(layer.weights <= 1)
