# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrowdingDistance from former test_nas.py

"""Focused suite: TestCrowdingDistance from former test_nas.py."""

from __future__ import annotations

from tests.nas_support import *  # noqa: F403

class TestCrowdingDistance:
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

    def test_two_points(self) -> None:
        front = [self._make_arch(0.9, 100), self._make_arch(0.8, 200)]
        dist = _crowding_distance(front)
        assert all(d == float("inf") for d in dist)

    def test_three_points(self) -> None:
        front = [
            self._make_arch(0.9, 300),
            self._make_arch(0.85, 200),
            self._make_arch(0.8, 100),
        ]
        dist = _crowding_distance(front)
        assert dist[0] == float("inf") or dist[2] == float("inf")
