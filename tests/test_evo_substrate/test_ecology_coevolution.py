# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCoevolution from former test_ecology.py

"""Focused suite: TestCoevolution from former test_ecology.py."""

from __future__ import annotations

from sc_neurocore.evo_substrate.ecology import (
    CoevolutionArena,
)
from sc_neurocore.evo_substrate.genome import (
    Genome,
)
from sc_neurocore.evo_substrate.organism import (
    Organism,
)


class TestCoevolution:
    """Organisms interacting in a coevolution arena."""

    def test_arena(self) -> None:
        """Two added organisms are counted in the arena."""
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
        """Two organisms give two interaction results."""
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
