# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHallOfFame from former test_selection.py

"""Focused suite: TestHallOfFame from former test_selection.py."""

from __future__ import annotations

from tests.test_evo_substrate.selection_support import *  # noqa: F403


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
