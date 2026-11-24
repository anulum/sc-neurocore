# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidation from former test_model_gamma_renewal.py

"""Focused suite: TestValidation from former test_model_gamma_renewal.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
import warnings
from sc_neurocore.neurons.models.gamma_renewal import (
    GammaRenewalNeuron,
)


class TestValidation:
    """Input and state validation of the GammaRenewalNeuron."""

    @pytest.mark.parametrize("rate_hz", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_baseline_rate(self, rate_hz: float) -> None:
        """The model rejects negative or non-finite baseline rate."""
        with pytest.raises(ValueError, match="rate_hz"):
            GammaRenewalNeuron(rate_hz=rate_hz)

    @pytest.mark.parametrize("shape_k", [0, -1, 1.5])
    def test_rejects_non_positive_or_non_integer_shape(self, shape_k: Any) -> None:
        """The model rejects non-positive or non-integer shape."""
        with pytest.raises(ValueError, match="shape_k"):
            GammaRenewalNeuron(shape_k=shape_k)

    @pytest.mark.parametrize("dt_ms", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_dt(self, dt_ms: float) -> None:
        """The model rejects non-positive or non-finite dt."""
        with pytest.raises(ValueError, match="dt_ms"):
            GammaRenewalNeuron(dt_ms=dt_ms)

    @pytest.mark.parametrize("_time_since_spike", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_elapsed_state(self, _time_since_spike: float) -> None:
        """The model rejects negative or non-finite elapsed state."""
        with pytest.raises(ValueError, match="time_since_spike"):
            GammaRenewalNeuron(_time_since_spike=_time_since_spike)

    @pytest.mark.parametrize("rate_override", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_rate_override_before_elapsed_mutation(
        self, rate_override: float
    ) -> None:
        """The model rejects non-finite rate override before elapsed mutation."""
        n = GammaRenewalNeuron(_time_since_spike=0.125)
        before = n._time_since_spike
        with pytest.raises(ValueError, match="rate_override"):
            n.step(rate_override=rate_override)
        assert n._time_since_spike == before

    def test_zero_rate_path_is_silent_and_never_spikes(self) -> None:
        """Zero rate path is silent and never spikes."""
        n = GammaRenewalNeuron(rate_hz=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            spikes = [n.step(rate_override=0.0) for _ in range(8)]
        assert spikes == [0] * 8
        assert n._time_since_spike > 0.0
