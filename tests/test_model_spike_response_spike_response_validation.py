# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeResponseValidation from former test_model_spike_response.py

"""Focused suite: TestSpikeResponseValidation from former test_model_spike_response.py."""

from __future__ import annotations

from tests.model_spike_response_support import *  # noqa: F403


class TestSpikeResponseValidation:
    @pytest.mark.parametrize("field", ["v", "v_threshold", "eta_reset"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SpikeResponseNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_eta", "tau_kappa", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_time_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SpikeResponseNeuron(**{field: value})

    @pytest.mark.parametrize("time_since_spike", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_refractory_clock(self, time_since_spike: float):
        with pytest.raises(ValueError, match="time_since_spike"):
            SpikeResponseNeuron(time_since_spike=time_since_spike)

    @pytest.mark.parametrize("weighted_input", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_input_before_state_mutation(self, weighted_input: float):
        n = SpikeResponseNeuron(v=0.25, time_since_spike=3.0)
        before = (n.v, n.time_since_spike)
        with pytest.raises(ValueError, match="weighted_input"):
            n.step(weighted_input)
        assert (n.v, n.time_since_spike) == before
