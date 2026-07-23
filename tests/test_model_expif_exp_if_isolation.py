# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFIsolation from former test_model_expif.py

"""Focused suite: TestExpIFIsolation from former test_model_expif.py."""

from __future__ import annotations

from tests.model_expif_support import *  # noqa: F403

class TestExpIFIsolation:
    def test_construction_uses_source_fitted_defaults(self) -> None:
        neuron = ExpIFNeuron()
        assert neuron.v == -65.0
        assert neuron.v_rest == -65.0
        assert neuron.v_reset == -68.0
        assert neuron.v_threshold == 30.0
        assert neuron.v_rh == -59.9
        assert neuron.delta_t == 3.48
        assert neuron.tau == 10.0
        assert neuron.dt == 0.02
        assert neuron.refractory_period == 0.0
        assert neuron.refractory_remaining == 0.0

    def test_step_returns_binary(self) -> None:
        assert ExpIFNeuron().step(0.0) in (0, 1)

    def test_state_evolves(self) -> None:
        neuron = ExpIFNeuron()
        initial = neuron.v
        neuron.step(20.0)
        assert neuron.v != initial

    def test_state_remains_finite_and_below_cutoff(self) -> None:
        neuron = ExpIFNeuron()
        for _ in range(50_000):
            neuron.step(20.0)
        assert math.isfinite(neuron.v)
        assert neuron.v < neuron.v_threshold

    def test_reset_restores_rest_and_clears_refractory_state(self) -> None:
        neuron = ExpIFNeuron(refractory_period=0.06)
        neuron.v = 29.0
        assert neuron.step(0.0) == 1
        assert neuron.refractory_remaining == pytest.approx(0.06)
        neuron.reset()
        assert neuron.v == neuron.v_rest
        assert neuron.refractory_remaining == 0.0
