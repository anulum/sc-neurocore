# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary speciation and diversity tests

"""Evolutionary speciation and diversity tests."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore.evo_substrate.speciation as speciation_mod
from sc_neurocore.evo_substrate.fitness import FitnessResult
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.evo_substrate.speciation import (
    assign_species,
    genomic_distance,
    population_diversity,
    shared_fitness,
)


class TestSpeciation:
    def test_identical_genomes_same_species(self) -> None:
        orgs = [Organism(genome=Genome()) for _ in range(5)]
        for o in orgs:
            o.genome.compute_id()
        species = assign_species(orgs, threshold=0.5)
        assert len(species) == 1

    def test_different_genomes_separate_species(self) -> None:
        orgs = []
        for i in range(3):
            g = Genome()
            g.topology.num_neurons = (i + 1) * 200
            g.neuron.tau_fast = (i + 1) * 50.0
            g.compute_id()
            orgs.append(Organism(genome=g))
        species = assign_species(orgs, threshold=0.01)
        assert len(species) >= 2

    def test_genomic_distance_self(self) -> None:
        g = Genome()
        assert genomic_distance(g, g) == 0.0

    def test_genomic_distance_symmetric(self) -> None:
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 100
        assert abs(genomic_distance(a, b) - genomic_distance(b, a)) < 1e-10

    def test_genomic_distance_numpy_fallback_matches_reference_formula(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(speciation_mod, "_HAS_RUST_EVO", False)
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 100
        va, vb = a.to_vector(), b.to_vector()
        expected = float(np.mean(np.abs(va - vb) / (np.abs(va) + np.abs(vb) + 1e-10)))

        assert genomic_distance(a, b) == pytest.approx(expected)


# ── Diversity Tests ─────────────────────────────────────────────────


class TestDiversity:
    def test_clones_zero_diversity(self) -> None:
        orgs = [Organism(genome=Genome()) for _ in range(5)]
        assert population_diversity(orgs) == 0.0

    def test_varied_population_positive_diversity(self) -> None:
        orgs = []
        for i in range(5):
            g = Genome()
            g.topology.num_neurons = 10 + i * 50
            orgs.append(Organism(genome=g))
        assert population_diversity(orgs) > 0.0

    def test_single_organism_zero(self) -> None:
        assert population_diversity([Organism(genome=Genome())]) == 0.0


# ── Lineage Tests ───────────────────────────────────────────────────


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


# ── CPPN Tests (Gap 16) ───────────────────────────────────────────────
