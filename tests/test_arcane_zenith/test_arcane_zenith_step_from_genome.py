# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStepFromGenome from former test_arcane_zenith.py

"""Focused suite: TestStepFromGenome from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403


class TestStepFromGenome:
    """``step_from_genome`` seeds ``tau_fast`` and ``tau_work`` from the
    ``NeuronGene`` fields of an evo_substrate Genome, and steps the
    cognitive core with ``genome.topology.connectivity`` as drive
    current. (``tau_deep`` is also seeded from the genome but is
    immediately overwritten by the plasticity sigmoid mapping later in
    the same call — by design, plasticity takes over after the genome
    seeds the initial scale.)
    """

    @pytest.fixture
    def genome(self):
        pytest.importorskip("sc_neurocore.evo_substrate.evo_substrate")
        from sc_neurocore.evo_substrate.evo_substrate import (
            Genome,
            NeuronGene,
            TopologyGene,
        )

        g = Genome()
        g.neuron = NeuronGene(tau_fast=7.5, tau_work=250.0, tau_deep=12500.0)
        g.topology = TopologyGene(connectivity=0.42)
        return g

    def test_seeds_tau_fast_and_tau_work_from_neuron_gene(self, genome):
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        core.step_from_genome(genome)
        # tau_fast and tau_work are seeded and NOT touched by plasticity
        # post-step (only tau_deep goes through the sigmoid map).
        assert core.neuron.tau_fast == pytest.approx(7.5)
        assert core.neuron.tau_work == pytest.approx(250.0)

    def test_advances_neuron_clock_once(self, genome):
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        steps_before = core.neuron.get_state()["total_steps"]
        core.step_from_genome(genome)
        assert core.neuron.get_state()["total_steps"] == steps_before + 1

    def test_tau_deep_stays_in_biological_range(self, genome):
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        core.step_from_genome(genome)
        # After step(), the sigmoid map clamps tau_deep into [1000, 50000]
        # regardless of whatever the genome seeded.
        assert 1000.0 <= core.neuron.tau_deep <= 50000.0

    def test_repeated_calls_keep_all_meta_params_bounded(self, genome):
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        for _ in range(50):
            core.step_from_genome(genome)
        assert 1000.0 <= core.neuron.tau_deep <= 50000.0
        assert 0.01 <= core.neuron.surprise_baseline <= 0.5
        assert 0.0 <= core.neuron.delta_conf <= 1.0
        assert 0.001 <= core.neuron.lr_base <= 0.1
