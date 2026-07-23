# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpeciation from former test_speciation.py

"""Focused suite: TestSpeciation from former test_speciation.py."""

from __future__ import annotations

from tests.test_evo_substrate.speciation_support import *  # noqa: F403

class TestSpeciation:
    def test_identical_genomes_same_species(self) -> None:
        orgs = [Organism(genome=Genome()) for _ in range(5)]
        for o in orgs:
            o.genome.compute_id()
        species = assign_species(orgs, threshold=0.5)
        assert len(species) == 1

    def test_different_genomes_separate_species(self) -> None:
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
        g = Genome()
        assert genomic_distance(g, g) == 0.0

    def test_genomic_distance_symmetric(self) -> None:
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 100
        assert abs(genomic_distance(a, b) - genomic_distance(b, a)) < 1e-10

    def test_genomic_distance_numpy_fallback_matches_reference_formula(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(speciation_mod, "_HAS_RUST_EVO", False)
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 100
        va, vb = a.to_vector(), b.to_vector()
        expected = float(np.mean(np.abs(va - vb) / (np.abs(va) + np.abs(vb) + 1e-10)))

        assert genomic_distance(a, b) == pytest.approx(expected)
