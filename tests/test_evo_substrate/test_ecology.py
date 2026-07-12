# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary population-ecology tests

"""Evolutionary population-ecology tests."""

from __future__ import annotations

import numpy as np

from sc_neurocore.evo_substrate.ecology import (
    CoevolutionArena,
    ExtinctionDetector,
    IslandModel,
    NoveltyArchive,
)
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism


class TestIslandModel:
    def test_add_organism(self) -> None:
        im = IslandModel(num_islands=3)
        g = Genome()
        g.compute_id()
        im.add_organism(0, Organism(genome=g))
        assert im.total_population == 1

    def test_migrate(self) -> None:
        im = IslandModel(num_islands=2, migration_rate=1.0)
        g = Genome()
        g.compute_id()
        im.add_organism(0, Organism(genome=g))
        rng = np.random.default_rng(42)
        im.migrate(rng)
        assert im.total_population >= 2  # original + migrant

    def test_single_island_migration_is_noop(self) -> None:
        im = IslandModel(num_islands=1, migration_rate=1.0)
        g = Genome()
        g.compute_id()
        im.add_organism(0, Organism(genome=g))

        assert im.migrate(np.random.default_rng(42)) == 0
        assert im.total_migrations == 0

    def test_zero_rate_skips_every_migration_attempt(self) -> None:
        im = IslandModel(num_islands=2, migration_rate=0.0)
        im.add_organism(0, Organism(genome=Genome()))

        assert im.migrate(np.random.default_rng(42)) == 0
        assert im.total_population == 1

    def test_selected_empty_islands_produce_no_migrants(self) -> None:
        im = IslandModel(num_islands=2, migration_rate=1.0)

        assert im.migrate(np.random.default_rng(42)) == 0
        assert im.total_population == 0


# ── Genome Serialization Tests (Gap 5) ───────────────────────────────


class TestNoveltyArchive:
    def test_empty_archive_high_score(self) -> None:
        na = NoveltyArchive()
        assert na.novelty_score(np.array([1.0, 2.0])) == 1.0

    def test_add_novel(self) -> None:
        na = NoveltyArchive(threshold=0.01)
        assert na.maybe_add(np.array([1.0, 0.0]))
        assert na.size == 1

    def test_add_duplicate_rejected(self) -> None:
        na = NoveltyArchive(threshold=0.5)
        na.maybe_add(np.array([1.0, 0.0]))
        assert not na.maybe_add(np.array([1.0, 0.0]))  # identical


# ── Resource Budget Tests (Gap 7) ─────────────────────────────────────


class TestExtinctionDetector:
    def test_no_extinction_early(self) -> None:
        ed = ExtinctionDetector(stagnation_gens=5)
        for i in range(3):
            assert ed.check(0.5) is False

    def test_detects_stagnation(self) -> None:
        ed = ExtinctionDetector(stagnation_gens=5)
        for _ in range(10):
            ed.check(0.5)  # all same fitness
        assert ed.extinction_count > 0

    def test_apply_kills(self) -> None:
        ed = ExtinctionDetector(kill_fraction=0.5)
        pop = [Organism(genome=Genome()) for _ in range(10)]
        rng = np.random.default_rng(42)
        killed = ed.apply(pop, rng)
        assert killed == 5

    def test_improving_history_does_not_trigger_extinction(self) -> None:
        ed = ExtinctionDetector(stagnation_gens=3)

        assert ed.check(0.1) is False
        assert ed.check(0.2) is False
        assert ed.check(0.3) is False
        assert ed.extinction_count == 0


# ── Co-Evolution Tests (Gap 9) ────────────────────────────────────────


class TestCoevolution:
    def test_arena(self) -> None:
        arena = CoevolutionArena()
        g1 = Genome()
        g1.topology.num_neurons = 32
        g1.compute_id()
        g2 = Genome()
        g2.topology.num_neurons = 8
        g2.compute_id()
        arena.add_predator(Organism(genome=g1))
        arena.add_prey(Organism(genome=g2))
        assert arena.total_organisms == 2

    def test_interactions(self) -> None:
        arena = CoevolutionArena()
        g1 = Genome()
        g1.topology.num_neurons = 32
        g1.compute_id()
        g2 = Genome()
        g2.topology.num_neurons = 8
        g2.compute_id()
        arena.add_predator(Organism(genome=g1))
        arena.add_prey(Organism(genome=g2))
        results = arena.evaluate_interactions()
        assert len(results) == 2


# ── Formal Safety Guard Tests (Gap 10) ────────────────────────────────
