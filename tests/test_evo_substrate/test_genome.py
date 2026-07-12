# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary genome contract tests

"""Evolutionary genome contract tests."""

from __future__ import annotations

import numpy as np

from sc_neurocore.evo_substrate.genome import (
    Genome,
    GenomeSerializer,
    NeuronGene,
    PlasticityGene,
    TopologyGene,
)


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


# ── TopologyGene Tests ───────────────────────────────────────────────


class TestTopologyGene:
    def test_from_vector_clamps(self) -> None:
        v = np.array([0.0, 0.0, -1.0, -1.0, 0.0])
        tg = TopologyGene.from_vector(v)
        assert tg.num_neurons >= 2
        assert tg.num_layers >= 1
        assert tg.connectivity >= 0.01
        assert tg.bitstream_length >= 32


# ── NeuronGene Tests ─────────────────────────────────────────────────


class TestNeuronGene:
    def test_from_vector_clamps(self) -> None:
        v = np.zeros(8)
        ng = NeuronGene.from_vector(v)
        assert ng.tau_fast >= 0.5
        assert ng.theta >= 0.1


# ── PlasticityGene Tests ─────────────────────────────────────────────


class TestPlasticityGene:
    def test_from_vector_clamps(self) -> None:
        v = np.zeros(6)
        pg = PlasticityGene.from_vector(v)
        assert pg.stdp_lr > 0
        assert pg.stp_u_base >= 0.01


# ── MutationEngine Tests ─────────────────────────────────────────────


class TestGenomeSerializer:
    def test_roundtrip(self) -> None:
        g = Genome()
        g.compute_id()
        d = GenomeSerializer.to_dict(g)
        g2 = GenomeSerializer.from_dict(d)
        assert g2.genome_id == g.genome_id
        np.testing.assert_array_almost_equal(g2.to_vector(), g.to_vector(), decimal=4)

    def test_dict_keys(self) -> None:
        g = Genome()
        g.compute_id()
        d = GenomeSerializer.to_dict(g)
        assert "vector" in d
        assert "genome_id" in d


# ── Novelty Search Tests (Gap 6) ──────────────────────────────────────
