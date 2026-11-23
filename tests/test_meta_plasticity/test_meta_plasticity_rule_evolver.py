# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRuleEvolver from former test_meta_plasticity.py

"""Focused suite: TestRuleEvolver from former test_meta_plasticity.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.meta_plasticity.meta_plasticity import (
    PlasticityRuleSet,
    RuleEvolver,
)


class TestRuleEvolver:
    """Evolution of plasticity rule sets."""

    def test_initial_population(self) -> None:
        """The evolver starts with its population size."""
        ev = RuleEvolver(population_size=8)
        assert len(ev.population) == 8

    def test_evaluate_fitness(self) -> None:
        """Evaluating a rule set records a positive fitness."""
        ev = RuleEvolver()
        rs = PlasticityRuleSet()
        fitness = ev.evaluate_fitness(rs, {"gci": 0.8, "gci_std": 0.05, "mean_surprise": 0.1})
        assert fitness > 0
        assert rs.fitness == fitness

    def test_crossover(self) -> None:
        """A child belongs to the next generation."""
        ev = RuleEvolver()
        p1 = PlasticityRuleSet()
        p2 = PlasticityRuleSet()
        p2.stdp.lr = 0.05
        child = ev.crossover(p1, p2)
        assert child.generation == ev.generation + 1

    def test_mutate(self) -> None:
        """Mutation changes a rule vector."""
        ev = RuleEvolver(mutation_rate=1.0, mutation_scale=0.5)
        original = PlasticityRuleSet()
        mutated = ev.mutate(original)
        v1 = original.to_vector()
        v2 = mutated.to_vector()
        assert not np.allclose(v1, v2)

    def test_evolve(self) -> None:
        """Evolving keeps the size and advances a generation."""
        ev = RuleEvolver(population_size=8)
        for r in ev.population:
            r.fitness = np.random.default_rng(42).random()
        new_pop = ev.evolve()
        assert len(new_pop) == 8
        assert ev.generation == 1

    def test_best(self) -> None:
        """The best member has the highest fitness."""
        ev = RuleEvolver(population_size=4)
        ev.population[0].fitness = 0.1
        ev.population[1].fitness = 0.9
        ev.population[2].fitness = 0.5
        ev.population[3].fitness = 0.3
        assert ev.best.fitness == 0.9

    def test_mean_fitness(self) -> None:
        """The mean fitness lies between zero and one."""
        ev = RuleEvolver(population_size=4)
        for i, r in enumerate(ev.population):
            r.fitness = float(i) / 3.0
        assert 0.0 < ev.mean_fitness < 1.0
