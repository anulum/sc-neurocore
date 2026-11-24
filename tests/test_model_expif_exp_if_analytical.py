# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFAnalytical from former test_model_expif.py

"""Focused suite: TestExpIFAnalytical from former test_model_expif.py."""

from __future__ import annotations

import pytest
from sc_neurocore.neurons.models.expif import (
    ExpIFNeuron,
)
from tests.model_expif_support import (
    _euler_candidate,
    _rk4_candidate,
)


class TestExpIFAnalytical:
    """Analytical behaviour of the ExpIFNeuron."""

    def test_one_step_matches_independent_rk4(self) -> None:
        """One step matches independent RK4."""
        neuron = ExpIFNeuron(v=-62.0, dt=0.02)
        expected = _rk4_candidate(neuron, 5.0)
        assert neuron.step(5.0) == 0
        assert neuron.v == pytest.approx(expected, abs=1.0e-12)

    def test_rk4_separates_from_raw_euler_near_onset(self) -> None:
        """RK4 separates from raw Euler near onset."""
        neuron = ExpIFNeuron(v=-56.0, dt=0.2)
        rk4 = _rk4_candidate(neuron, 12.0)
        euler = _euler_candidate(neuron, 12.0)
        assert abs(rk4 - euler) > 1.0e-4
        assert neuron.step(12.0) == 0
        assert neuron.v == pytest.approx(rk4, abs=1.0e-12)

    def test_zero_current_relaxes_to_source_equilibrium(self) -> None:
        """Zero current relaxes to source equilibrium."""
        neuron = ExpIFNeuron()
        for _ in range(10_000):
            neuron.step(0.0)
        assert abs(neuron.v - neuron.v_rest) < 1.2
        assert neuron.v == pytest.approx(-63.896297890416314, abs=1.0e-10)

    def test_refractory_hold_is_discrete_and_deterministic(self) -> None:
        """Refractory hold is discrete and deterministic."""
        neuron = ExpIFNeuron(v=29.0, refractory_period=0.06)
        assert neuron.step(0.0) == 1
        for _ in range(3):
            assert neuron.step(100.0) == 0
            assert neuron.v == neuron.v_reset
        assert neuron.refractory_remaining == 0.0
        neuron.step(100.0)
        assert neuron.v != neuron.v_reset
