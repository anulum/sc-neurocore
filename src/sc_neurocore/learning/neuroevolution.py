# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Genetic Algorithm for evolving SNN weights/parameters

"""Genetic algorithm for evolving spiking-neural-network weights and parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SNNGeneticEvolver:
    """Genetic algorithm for evolving SNN weights and parameters."""

    population_size: int = 20
    mutation_rate: float = 0.05
    elite_fraction: float = 0.2

    def __init__(self, layer_factory: Callable[[], Any], fitness_func: Callable[[Any], float]):
        self.layer_factory = layer_factory
        self.fitness_func = fitness_func
        # Initialize population
        self.population = [layer_factory() for _ in range(self.population_size)]

    def evolve(self, generations: int) -> Any:
        """Run the GA for the given number of generations and return the best individual."""
        for gen in range(generations):
            # 1. Evaluate Fitness
            scores = [self.fitness_func(ind) for ind in self.population]

            # Sort by fitness (descending)
            ranked_indices = np.argsort(scores)[::-1]
            ranked_pop = [self.population[i] for i in ranked_indices]

            logger.info("Gen %d: Best Fitness = %.4f", gen, scores[ranked_indices[0]])

            # 2. Selection (Elitism)
            n_elite = int(self.population_size * self.elite_fraction)
            next_gen = ranked_pop[:n_elite]

            # 3. Crossover & Mutation
            while len(next_gen) < self.population_size:
                # Simple random selection for parents
                p1, p2 = np.random.choice(ranked_pop[: n_elite + 5], 2, replace=False)
                child = self._crossover(p1, p2)
                self._mutate(child)
                next_gen.append(child)

            self.population = next_gen

        return self.population[0]  # Return best

    def _crossover(self, p1: Any, p2: Any) -> Any:
        # Create new instance
        child = self.layer_factory()
        if not hasattr(p1, "weights"):
            return child

        # Uniform crossover
        mask = np.random.rand(*p1.weights.shape) > 0.5
        child.weights = np.where(mask, p1.weights, p2.weights)
        return child

    def _mutate(self, ind: Any) -> None:
        if not hasattr(ind, "weights"):
            return

        # Gaussian mutation
        mutation_mask = np.random.rand(*ind.weights.shape) < self.mutation_rate
        noise = np.random.normal(0, 0.1, ind.weights.shape)
        ind.weights[mutation_mask] += noise[mutation_mask]
        ind.weights = np.clip(ind.weights, 0, 1)
