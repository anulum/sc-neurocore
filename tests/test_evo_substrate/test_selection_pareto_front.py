# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParetoFront from former test_selection.py

"""Focused suite: TestParetoFront from former test_selection.py."""

from __future__ import annotations

from tests.test_evo_substrate.selection_support import *  # noqa: F403


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
