# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossover from former test_variation.py

"""Focused suite: TestCrossover from former test_variation.py."""

from __future__ import annotations

from tests.test_evo_substrate.variation_support import *  # noqa: F403


class TestCrossover:
    def test_crossover_produces_child(self) -> None:
        cx = CrossoverEngine(rng_seed=42)
        a = Genome()
        a.compute_id()
        b = Genome()
        b.topology.num_neurons = 64
        b.compute_id()
        child = cx.crossover(a, b)
        assert child.genome_id != ""
        assert child.generation == 1

    def test_crossover_parent_id_format(self) -> None:
        cx = CrossoverEngine()
        a = Genome()
        a.compute_id()
        b = Genome()
        b.compute_id()
        child = cx.crossover(a, b)
        assert "x" in child.parent_id

    def test_crossover_mixes_genes(self) -> None:
        cx = CrossoverEngine(rng_seed=7)
        a = Genome()
        a.neuron.tau_fast = 1.0
        a.compute_id()
        b = Genome()
        b.neuron.tau_fast = 100.0
        b.compute_id()
        child = cx.crossover(a, b)
        assert child.neuron.tau_fast in (1.0, 100.0)  # mix
