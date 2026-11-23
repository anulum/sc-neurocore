# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpeciation from former test_speciation.py

"""Focused suite: TestSpeciation from former test_speciation.py."""

from __future__ import annotations

import numpy as np
import pytest
import sc_neurocore.evo_substrate.speciation as speciation_mod
from sc_neurocore.evo_substrate.genome import (
    Genome,
)
from sc_neurocore.evo_substrate.organism import (
    Organism,
)
from sc_neurocore.evo_substrate.speciation import (
    assign_species,
    genomic_distance,
)


class TestSpeciation:
    """Grouping genomes into species."""

    def test_identical_genomes_same_species(self) -> None:
        """Identical genomes form one species."""
        orgs = [Organism(genome=Genome()) for _ in range(5)]
        for o in orgs:
            o.genome.compute_id()
        species = assign_species(orgs, threshold=0.5)
        assert len(species) == 1

    def test_different_genomes_separate_species(self) -> None:
        """Different genomes form at least two species."""
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
        """A genome is at distance 0 from itself."""
        g = Genome()
        assert genomic_distance(g, g) == 0.0

    def test_genomic_distance_symmetric(self) -> None:
        """The genomic distance is symmetric."""
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 100
        assert abs(genomic_distance(a, b) - genomic_distance(b, a)) < 1e-10

    def test_genomic_distance_numpy_fallback_matches_reference_formula(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The NumPy fallback matches the reference distance."""
        monkeypatch.setattr(speciation_mod, "_HAS_RUST_EVO", False)
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 100
        va, vb = a.to_vector(), b.to_vector()
        expected = float(np.mean(np.abs(va - vb) / (np.abs(va) + np.abs(vb) + 1e-10)))

        assert genomic_distance(a, b) == pytest.approx(expected)
