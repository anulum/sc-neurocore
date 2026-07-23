# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIslandModel from former test_ecology.py

"""Focused suite: TestIslandModel from former test_ecology.py."""

from __future__ import annotations

from tests.test_evo_substrate.ecology_support import *  # noqa: F403

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
