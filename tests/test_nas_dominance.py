# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDominance from former test_nas.py

"""Focused suite: TestDominance from former test_nas.py."""

from __future__ import annotations

from tests.nas_support import *  # noqa: F403


class TestDominance:
    def test_dominates(self) -> None:
        a = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=0.9,
            fitness_energy_nj=100.0,
        )
        b = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=0.8,
            fitness_energy_nj=200.0,
        )
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_no_dominance(self) -> None:
        a = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=0.9,
            fitness_energy_nj=200.0,
        )
        b = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=0.8,
            fitness_energy_nj=100.0,
        )
        assert not _dominates(a, b)
        assert not _dominates(b, a)
