# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFLIFValidation from former test_model_fractional_lif.py

"""Focused suite: TestFLIFValidation from former test_model_fractional_lif.py."""

from __future__ import annotations

from tests.model_fractional_lif_support import *  # noqa: F403

class TestFLIFValidation:
    @pytest.mark.parametrize("field", ["v", "v_rest", "v_reset", "v_threshold"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            FractionalLIFNeuron(**{field: value})

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, np.nan, np.inf])
    def test_rejects_fractional_order_outside_open_closed_unit_interval(self, alpha: float):
        with pytest.raises(ValueError, match="alpha"):
            FractionalLIFNeuron(alpha=alpha)

    @pytest.mark.parametrize("field", ["resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            FractionalLIFNeuron(**{field: value})

    @pytest.mark.parametrize("_max_history", [0, -1, 1.5])
    def test_rejects_non_positive_or_non_integer_history_length(self, _max_history):
        with pytest.raises(ValueError, match="max_history"):
            FractionalLIFNeuron(_max_history=_max_history)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_or_history_mutation(self, current: float):
        n = FractionalLIFNeuron(v=0.25)
        before = (n.v, list(n._history))
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n._history) == before
