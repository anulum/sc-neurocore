# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCFCValidation from former test_model_cfc.py

"""Focused suite: TestCFCValidation from former test_model_cfc.py."""

from __future__ import annotations

from tests.model_cfc_support import *  # noqa: F403

class TestCFCValidation:
    @pytest.mark.parametrize("field", ["x", "w_tau", "w_x", "w_in", "bias"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_weights(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            ClosedFormContinuousNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_base", "v_threshold", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            ClosedFormContinuousNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = ClosedFormContinuousNeuron(x=0.25)
        before = n.x
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.x == before
