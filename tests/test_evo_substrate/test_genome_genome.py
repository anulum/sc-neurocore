# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGenome from former test_genome.py

"""Focused suite: TestGenome from former test_genome.py."""

from __future__ import annotations

from tests.test_evo_substrate.genome_support import *  # noqa: F403


class TestGenome:
    def test_to_vector(self) -> None:
        g = Genome()
        v = g.to_vector()
        assert len(v) == g.vector_dim

    def test_from_vector_roundtrip(self) -> None:
        g = Genome()
        v = g.to_vector()
        g2 = Genome.from_vector(v)
        assert abs(g2.topology.num_neurons - g.topology.num_neurons) < 1
        assert abs(g2.neuron.tau_fast - g.neuron.tau_fast) < 0.01

    def test_compute_id(self) -> None:
        g = Genome()
        gid = g.compute_id()
        assert len(gid) == 12
        assert g.genome_id == gid

    def test_id_deterministic(self) -> None:
        g1 = Genome()
        g2 = Genome()
        assert g1.compute_id() == g2.compute_id()

    def test_id_differs_on_change(self) -> None:
        g1 = Genome()
        g2 = Genome()
        g2.topology.num_neurons = 999
        assert g1.compute_id() != g2.compute_id()
