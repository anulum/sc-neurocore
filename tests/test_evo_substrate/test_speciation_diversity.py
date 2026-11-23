# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiversity from former test_speciation.py

"""Focused suite: TestDiversity from former test_speciation.py."""

from __future__ import annotations

from sc_neurocore.evo_substrate.genome import (
    Genome,
)
from sc_neurocore.evo_substrate.organism import (
    Organism,
)
from sc_neurocore.evo_substrate.speciation import (
    population_diversity,
)


class TestDiversity:
    """Population diversity."""

    def test_clones_zero_diversity(self) -> None:
        """A population of clones has no diversity."""
        orgs = [Organism(genome=Genome()) for _ in range(5)]
        assert population_diversity(orgs) == 0.0

    def test_varied_population_positive_diversity(self) -> None:
        """A varied population has positive diversity."""
        orgs = []
        for i in range(5):
            g = Genome()
            g.topology.num_neurons = 10 + i * 50
            orgs.append(Organism(genome=g))
        assert population_diversity(orgs) > 0.0

    def test_single_organism_zero(self) -> None:
        """A single organism has no diversity."""
        assert population_diversity([Organism(genome=Genome())]) == 0.0
