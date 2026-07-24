# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMutationEngine from former test_variation.py

"""Focused suite: TestMutationEngine from former test_variation.py."""

from __future__ import annotations

from tests.test_evo_substrate.variation_support import *  # noqa: F403


class TestMutationEngine:
    def test_point_mutation(self) -> None:
        me = MutationEngine(
            MutationConfig(point_rate=1.0, structural_rate=0, duplication_rate=0, swap_rate=0)
        )
        g = Genome()
        g.compute_id()
        child, mt = me.mutate(g)
        assert mt == MutationType.POINT
        assert child.parent_id == g.genome_id
        assert child.generation == g.generation + 1

    def test_structural_mutation(self) -> None:
        me = MutationEngine(MutationConfig(structural_rate=1.0), rng_seed=0)
        g = Genome()
        g.compute_id()
        child, mt = me.mutate(g)
        assert mt == MutationType.STRUCTURAL

    def test_child_has_new_id(self) -> None:
        me = MutationEngine()
        g = Genome()
        g.compute_id()
        child, _ = me.mutate(g)
        assert child.genome_id != ""

    def test_child_identity_reset(self) -> None:
        me = MutationEngine()
        g = Genome()
        g.identity_deep = 0.99
        g.compute_id()
        child, _ = me.mutate(g)
        assert child.identity_deep == 0.0

    def test_neuron_bounds_preserved(self) -> None:
        me = MutationEngine(MutationConfig(point_rate=1.0, point_sigma=10.0), rng_seed=42)
        g = Genome()
        g.compute_id()
        for _ in range(10):
            g, _ = me.mutate(g)
        assert g.neuron.tau_fast >= 0.5
        assert g.neuron.theta >= 0.1
        assert g.topology.num_neurons >= 2

    def test_duplication_mutation_expands_layers_and_neuron_budget(self) -> None:
        me = MutationEngine(
            MutationConfig(
                point_rate=0.0,
                structural_rate=0.0,
                duplication_rate=1.0,
                swap_rate=0.0,
                max_neurons=30,
            )
        )
        g = Genome()
        g.topology.num_neurons = 20
        g.topology.num_layers = 2
        g.compute_id()

        child, mutation_type = me.mutate(g)

        assert mutation_type == MutationType.DUPLICATION
        assert child.topology.num_layers == 3
        assert child.topology.num_neurons == 30
        assert child.parent_id == g.genome_id

    def test_swap_mutation_exchanges_fast_and_work_time_constants(self) -> None:
        me = MutationEngine(
            MutationConfig(
                point_rate=0.0,
                structural_rate=0.0,
                duplication_rate=0.0,
                swap_rate=1.0,
            )
        )
        g = Genome()
        g.neuron.tau_fast = 3.0
        g.neuron.tau_work = 41.0
        g.compute_id()

        child, mutation_type = me.mutate(g)

        assert mutation_type == MutationType.SWAP
        assert child.neuron.tau_fast == 41.0
        assert child.neuron.tau_work == 3.0
