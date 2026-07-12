# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary selection and survivor-regulation tests

"""Evolutionary selection and survivor-regulation tests."""

from __future__ import annotations

import numpy as np

from sc_neurocore.evo_substrate.fitness import FitnessResult
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.evo_substrate.selection import (
    AgeRegulator,
    BloatPenalizer,
    HallOfFame,
    ParetoFront,
    TournamentSelector,
    compute_bloat,
)


class TestHallOfFame:
    def test_update(self) -> None:
        hof = HallOfFame(max_size=3)
        g = Genome()
        g.compute_id()
        org = Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=0.8))
        assert hof.update(org)
        assert hof.best_fitness == 0.8

    def test_max_size(self) -> None:
        hof = HallOfFame(max_size=2)
        for i in range(5):
            g = Genome()
            g.topology.num_neurons = i + 10
            g.compute_id()
            org = Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=i * 0.1))
            hof.update(org)
        assert hof.size == 2

    def test_update_rejects_unevaluated_organism(self) -> None:
        hof = HallOfFame()
        assert hof.update(Organism(genome=Genome())) is False
        assert hof.size == 0


# ── Island Model Tests (Gap 4) ────────────────────────────────────────


class TestTournamentSelector:
    def test_select(self) -> None:
        ts = TournamentSelector(tournament_size=2)
        pop = []
        for i in range(5):
            g = Genome()
            g.topology.num_neurons = i + 10
            g.compute_id()
            org = Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=i * 0.1))
            pop.append(org)
        rng = np.random.default_rng(42)
        winner = ts.select(pop, rng)
        assert winner is not None

    def test_select_n(self) -> None:
        ts = TournamentSelector(tournament_size=3)
        pop = []
        for i in range(10):
            g = Genome()
            g.topology.num_neurons = i + 5
            g.compute_id()
            pop.append(Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=i * 0.05)))
        rng = np.random.default_rng(0)
        selected = ts.select_n(pop, 4, rng)
        assert len(selected) == 4


# ── Pareto Front Tests (Gap 12) ───────────────────────────────────────


class TestParetoFront:
    def test_add_non_dominated(self) -> None:
        pf = ParetoFront()
        g = Genome()
        g.compute_id()
        org = Organism(
            genome=g,
            fitness=FitnessResult(g.genome_id, accuracy=0.9, energy_score=0.5, latency_score=0.8),
        )
        assert pf.update(org)
        assert pf.size == 1

    def test_dominated_rejected(self) -> None:
        pf = ParetoFront()
        g1 = Genome()
        g1.compute_id()
        org1 = Organism(
            genome=g1,
            fitness=FitnessResult(g1.genome_id, accuracy=0.9, energy_score=0.9, latency_score=0.9),
        )
        pf.update(org1)
        g2 = Genome()
        g2.topology.num_neurons = 8
        g2.compute_id()
        org2 = Organism(
            genome=g2,
            fitness=FitnessResult(g2.genome_id, accuracy=0.5, energy_score=0.5, latency_score=0.5),
        )
        assert not pf.update(org2)

    def test_unevaluated_organism_is_not_added(self) -> None:
        pf = ParetoFront()
        assert pf.update(Organism(genome=Genome())) is False
        assert pf.size == 0


# ── Age Regulation Tests (Gap 13) ─────────────────────────────────────


class TestAgeRegulator:
    def test_young_survive(self) -> None:
        ar = AgeRegulator(max_age=10)
        pop = [Organism(genome=Genome(), birth_generation=5)]
        killed = ar.apply(pop, current_generation=10)
        assert killed == 0

    def test_old_culled(self) -> None:
        ar = AgeRegulator(max_age=5)
        pop = [Organism(genome=Genome(), birth_generation=0)]
        killed = ar.apply(pop, current_generation=10)
        assert killed == 1
        assert not pop[0].alive


# ── Bloat Control Tests (Gap 14) ──────────────────────────────────────


class TestBloatControl:
    def test_compute_bloat(self) -> None:
        g = Genome()
        bm = compute_bloat(g)
        assert bm.total_params > 0
        assert bm.bloat_score > 0

    def test_penalizer_no_penalty(self) -> None:
        bp = BloatPenalizer(threshold=100.0)
        g = Genome()
        assert bp.penalize(0.9, g) == 0.9

    def test_penalizer_reduces(self) -> None:
        bp = BloatPenalizer(threshold=0.01)
        g = Genome()
        assert bp.penalize(0.9, g) < 0.9

    def test_bloat_metrics_marks_large_genome_bloated(self) -> None:
        g = Genome()
        g.topology.num_neurons = 512
        g.topology.num_layers = 8
        metrics = compute_bloat(g, baseline_neurons=4)
        assert metrics.is_bloated


# ── Fitness Sharing Tests (Gap 15) ────────────────────────────────────
