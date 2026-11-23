# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTournamentSelector from former test_selection.py

"""Focused suite: TestTournamentSelector from former test_selection.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.evo_substrate.fitness import (
    FitnessResult,
)
from sc_neurocore.evo_substrate.genome import (
    Genome,
)
from sc_neurocore.evo_substrate.organism import (
    Organism,
)
from sc_neurocore.evo_substrate.selection import (
    TournamentSelector,
)


class TestTournamentSelector:
    """Tournament selection of parents."""

    def test_select(self) -> None:
        """A tournament returns a winner."""
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
        """Selecting four returns four organisms."""
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
