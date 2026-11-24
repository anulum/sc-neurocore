# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSNNGeneticEvolver from former test_research_modules.py

"""Focused suite: TestSNNGeneticEvolver from former test_research_modules.py."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.learning.neuroevolution import SNNGeneticEvolver
import numpy as np


class TestSNNGeneticEvolver:
    """Genetic evolution of spiking layers."""

    def _make_layer(self) -> Any:
        class MockLayer:
            """A layer stand-in with random 3x3 weights."""

            def __init__(self) -> None:
                self.weights = np.random.rand(3, 3)

        return MockLayer()

    def test_construction(self) -> None:
        """A new evolver holds a population of 20."""
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        assert len(evolver.population) == 20

    def test_evolve_returns_best(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Evolution returns a layer with weights."""
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        best = evolver.evolve(generations=3)
        assert hasattr(best, "weights")

    def test_crossover(self) -> None:
        """Crossover gives a child with 3x3 weights."""
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        p1 = self._make_layer()
        p2 = self._make_layer()
        child = evolver._crossover(p1, p2)
        assert child.weights.shape == (3, 3)

    def test_mutate(self) -> None:
        """Mutation changes the weights and keeps them inside [0, 1]."""
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
