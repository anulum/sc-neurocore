# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNonDominatedSort from former test_nas.py

"""Focused suite: TestNonDominatedSort from former test_nas.py."""

from __future__ import annotations

from tests.nas_support import *  # noqa: F403

class TestNonDominatedSort:
    def _make_arch(self, acc: float, energy: float) -> Architecture:
        return Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=acc,
            fitness_energy_nj=energy,
        )

    def test_single_front(self) -> None:
        pop = [self._make_arch(0.9, 200), self._make_arch(0.8, 100)]
        fronts = _non_dominated_sort(pop)
        assert len(fronts) == 1
        assert len(fronts[0]) == 2

    def test_two_fronts(self) -> None:
        pop = [
            self._make_arch(0.9, 100),  # dominates c
            self._make_arch(0.8, 50),  # dominates c
            self._make_arch(0.7, 200),  # dominated by both
        ]
        fronts = _non_dominated_sort(pop)
        assert len(fronts) == 2
        assert len(fronts[0]) == 2
        assert len(fronts[1]) == 1
