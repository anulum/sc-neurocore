# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTraubMilesParameters from former test_model_traub_miles.py

"""Focused suite: TestTraubMilesParameters from former test_model_traub_miles.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.traub_miles import (
    TraubMilesNeuron,
)
from tests.model_traub_miles_support import (
    _run,
)


class TestTraubMilesParameters:
    """Parameter behaviour of the TraubMilesNeuron."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("m", np.inf),
            ("h", -0.1),
            ("n", 1.1),
            ("g_na", -1.0),
            ("g_k", -1.0),
            ("g_l", -1.0),
            ("dt", 0.0),
            ("v_threshold", np.inf),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float) -> None:
        """An invalid numerical configuration is rejected."""
        with pytest.raises((ValueError, FloatingPointError)):
            kwargs: dict[str, Any] = {field: value}
            TraubMilesNeuron(**kwargs)

    def test_rejects_non_finite_current_before_state_mutation(self) -> None:
        """A non-finite current is rejected before any state change."""
        n = TraubMilesNeuron()
        before = (n.v, n.m, n.h, n.n)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.v, n.m, n.h, n.n) == before

    def test_rejects_corrupted_gate_before_state_mutation(self) -> None:
        """The model rejects corrupted gate before state mutation."""
        n = TraubMilesNeuron()
        n.m = 1.5
        before = (n.v, n.m, n.h, n.n)
        with pytest.raises(FloatingPointError, match="m gate"):
            n.step(5.0)
        assert (n.v, n.m, n.h, n.n) == before

    def test_rejects_rate_overflow_before_state_mutation(self) -> None:
        """The model rejects rate overflow before state mutation."""
        n = TraubMilesNeuron(v=-1.0e6)
        before = (n.v, n.m, n.h, n.n)
        with pytest.raises(FloatingPointError, match="rate evaluation"):
            n.step(5.0)
        assert (n.v, n.m, n.h, n.n) == before

    def test_rejects_corrupted_voltage_configuration_before_state_mutation(self) -> None:
        """The model rejects corrupted voltage configuration before state mutation."""
        n = TraubMilesNeuron()
        n.v = np.nan
        before = (n.v, n.m, n.h, n.n)
        with pytest.raises(ValueError, match="v must be finite"):
            n.step(5.0)
        actual = (n.v, n.m, n.h, n.n)
        assert np.isnan(actual[0])
        assert actual[1:] == before[1:]

    def test_state_kernel_rejects_non_finite_voltage(self) -> None:
        """State kernel rejects non-finite voltage."""
        with pytest.raises(FloatingPointError, match="voltage state"):
            TraubMilesNeuron._validate_state(float("nan"), 0.05, 0.6, 0.3)

    def test_rejects_non_finite_rate_kernel_input(self) -> None:
        """The model rejects non-finite rate kernel input."""
        with pytest.raises(FloatingPointError, match="rates"):
            TraubMilesNeuron._rates(float("nan"))

    def test_derivative_kernel_rejects_non_finite_current_balance(self) -> None:
        """Derivative kernel rejects non-finite current balance."""
        n = TraubMilesNeuron(g_na=1.0e308)
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(-65.0, 1.0, 1.0, 0.3, 0.0)

    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = TraubMilesNeuron(dt=dt)
        for _ in range(20000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_g_na_controls_excitability(self) -> None:
        """G Na controls excitability."""
        n_low = TraubMilesNeuron(g_na=50.0)
        n_high = TraubMilesNeuron(g_na=150.0)
        s_low = len(_run(n_low, current=5.0, steps=50000))
        s_high = len(_run(n_high, current=5.0, steps=50000))
        assert s_low != s_high

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = TraubMilesNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
