# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAgeRegulator from former test_selection.py

"""Focused suite: TestAgeRegulator from former test_selection.py."""

from __future__ import annotations

from sc_neurocore.evo_substrate.genome import (
    Genome,
)
from sc_neurocore.evo_substrate.organism import (
    Organism,
)
from sc_neurocore.evo_substrate.selection import (
    AgeRegulator,
)


class TestAgeRegulator:
    """Culling organisms by age."""

    def test_young_survive(self) -> None:
        """Young organisms survive."""
        ar = AgeRegulator(max_age=10)
        pop = [Organism(genome=Genome(), birth_generation=5)]
        killed = ar.apply(pop, current_generation=10)
        assert killed == 0

    def test_old_culled(self) -> None:
        """An old organism is culled."""
        ar = AgeRegulator(max_age=5)
        pop = [Organism(genome=Genome(), birth_generation=0)]
        killed = ar.apply(pop, current_generation=10)
        assert killed == 1
        assert not pop[0].alive
