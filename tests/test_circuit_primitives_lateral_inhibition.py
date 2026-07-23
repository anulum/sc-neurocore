# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLateralInhibition from former test_circuit_primitives.py

"""Focused suite: TestLateralInhibition from former test_circuit_primitives.py."""

from __future__ import annotations

from tests.circuit_primitives_support import *  # noqa: F403

class TestLateralInhibition:
    def test_suppresses_neighbors(self):
        li = LateralInhibition(n_neurons=5, inhibition_strength=0.5, radius=1)
        rates = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = li.apply(rates)
        assert result[2] == 1.0  # center untouched (no self-inhibition)
        assert result[1] == 0.0  # neighbor suppressed
        assert result[3] == 0.0  # neighbor suppressed
        assert result[0] == 0.0  # beyond radius

    def test_preserves_zero_input(self):
        li = LateralInhibition(n_neurons=10, inhibition_strength=0.5, radius=3)
        rates = np.zeros(10)
        result = li.apply(rates)
        np.testing.assert_allclose(result, 0.0)

    def test_circular_topology(self):
        li = LateralInhibition(n_neurons=6, inhibition_strength=0.5, radius=1)
        # Neuron 0 should inhibit neuron 5 (circular neighbor)
        rates = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        result = li.apply(rates)
        assert result[5] < 0.5  # inhibited by neuron 0

    def test_output_non_negative(self):
        li = LateralInhibition(n_neurons=10, inhibition_strength=1.0, radius=5)
        rates = np.random.rand(10)
        result = li.apply(rates)
        assert np.all(result >= 0.0)
