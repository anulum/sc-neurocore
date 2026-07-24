# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSearchSpace from former test_nas.py

"""Focused suite: TestSearchSpace from former test_nas.py."""

from __future__ import annotations

from tests.nas_support import *  # noqa: F403


class TestSearchSpace:
    def test_defaults(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10)
        assert sp.min_layers == 1
        assert sp.max_layers == 4
        assert len(sp.width_choices) == len(WIDTH_CHOICES)

    def test_random_architecture(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10)
        rng = np.random.RandomState(42)
        arch = sp.random_architecture(rng)
        assert arch.n_inputs == 64
        assert arch.layer_widths[-1] == 10
        assert arch.n_layers >= 1
        assert arch.n_layers <= 4
        for nt in arch.neuron_types:
            assert nt in NEURON_CHOICES
        for L in arch.bitstream_lengths:
            assert L in L_CHOICES

    def test_mutate(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10)
        rng = np.random.RandomState(42)
        original = sp.random_architecture(rng)
        mutated = sp.mutate(original, rng)
        assert mutated.n_inputs == original.n_inputs
        assert mutated.n_layers == original.n_layers

    def test_mutate_all_genes(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10, min_layers=3, max_layers=3)
        # Run enough mutations to hit all gene types including width (gene=0)
        for seed in range(50):
            rng = np.random.RandomState(seed)
            original = sp.random_architecture(np.random.RandomState(0))
            sp.mutate(original, rng)

    def test_crossover_same_layers(self) -> None:
        sp = SearchSpace(n_inputs=32, n_outputs=8, min_layers=2, max_layers=2)
        rng = np.random.RandomState(42)
        a = sp.random_architecture(rng)
        b = sp.random_architecture(rng)
        child = sp.crossover(a, b, rng)
        assert child.n_layers == 2
        assert child.layer_widths[-1] == 8

    def test_crossover_different_layers(self) -> None:
        sp = SearchSpace(n_inputs=32, n_outputs=8)
        rng = np.random.RandomState(42)
        a = Architecture(
            n_inputs=32,
            layer_widths=[16, 8],
            neuron_types=["StochasticLIFNeuron"] * 2,
            bitstream_lengths=[128, 64],
            delay_ranges=[0, 0],
        )
        b = Architecture(
            n_inputs=32,
            layer_widths=[32, 16, 8],
            neuron_types=["SCIzhikevichNeuron"] * 3,
            bitstream_lengths=[256, 128, 64],
            delay_ranges=[1, 1, 1],
        )
        child = sp.crossover(a, b, rng)
        assert child.n_layers == 2  # min(2, 3)

    def test_space_size(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10, min_layers=1, max_layers=2)
        assert sp.space_size > 0
