# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFAValidation from former test_model_sfa.py

"""Focused suite: TestSFAValidation from former test_model_sfa.py."""

from __future__ import annotations

from tests.model_sfa_support import *  # noqa: F403

class TestSFAValidation:
    @pytest.mark.parametrize("field", ["v", "v_rest", "v_reset", "v_threshold", "e_k"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SFANeuron(**{field: value})

    @pytest.mark.parametrize("g_sfa", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_adaptation_conductance(self, g_sfa: float):
        with pytest.raises(ValueError, match="g_sfa"):
            SFANeuron(g_sfa=g_sfa)

    @pytest.mark.parametrize("field", ["tau_m", "tau_sfa", "resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SFANeuron(**{field: value})

    @pytest.mark.parametrize("delta_g", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_spike_adaptation_increment(self, delta_g: float):
        with pytest.raises(ValueError, match="delta_g"):
            SFANeuron(delta_g=delta_g)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = SFANeuron(v=-65.0, g_sfa=0.25)
        before = (n.v, n.g_sfa)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.g_sfa) == before
