# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFitnessSharing from former test_speciation.py

"""Focused suite: TestFitnessSharing from former test_speciation.py."""

from __future__ import annotations

from tests.test_evo_substrate.speciation_support import *  # noqa: F403

class TestFitnessSharing:
    def test_shared_fitness_reduces(self) -> None:
        pop = []
        for _ in range(5):
            g = Genome()
            g.compute_id()
            pop.append(Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=0.8)))
        sf = shared_fitness(pop[0], pop, sigma=1.0)
        assert sf < 0.8  # shared among 5 clones

    def test_unique_keeps_full(self) -> None:
        g1 = Genome()
        g1.topology.num_neurons = 4
        g1.compute_id()
        g2 = Genome()
        g2.topology.num_neurons = 1000
        g2.compute_id()
        org1 = Organism(genome=g1, fitness=FitnessResult(g1.genome_id, composite=0.8))
        org2 = Organism(genome=g2, fitness=FitnessResult(g2.genome_id, composite=0.5))
        sf = shared_fitness(org1, [org1, org2], sigma=0.0001)
        assert sf > 0.5  # only shares with itself

    def test_unevaluated_organism_has_zero_shared_fitness(self) -> None:
        assert shared_fitness(Organism(genome=Genome()), [Organism(genome=Genome())]) == 0.0
