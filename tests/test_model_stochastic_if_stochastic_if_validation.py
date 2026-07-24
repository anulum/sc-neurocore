# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticIFValidation from former test_model_stochastic_if.py

"""Focused suite: TestStochasticIFValidation from former test_model_stochastic_if.py."""

from __future__ import annotations

from tests.model_stochastic_if_support import *  # noqa: F403


class TestStochasticIFValidation:
    @pytest.mark.parametrize("field", ["v", "v_rest", "v_reset", "v_threshold", "mu"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_and_drive_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            StochasticIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_m", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_timescales(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            StochasticIFNeuron(**{field: value})

    @pytest.mark.parametrize("sigma", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_noise_scale(self, sigma: float):
        with pytest.raises(ValueError, match="sigma"):
            StochasticIFNeuron(sigma=sigma)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        neuron = StochasticIFNeuron(v=-60.0)
        before = neuron.v
        with pytest.raises(ValueError, match="current"):
            neuron.step(current)
        assert neuron.v == before
