# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFValidation from former test_model_expif.py

"""Focused suite: TestExpIFValidation from former test_model_expif.py."""

from __future__ import annotations

from typing import Any
import math
import pytest
from sc_neurocore.neurons.models.expif import (
    ExpIFNeuron,
)


class TestExpIFValidation:
    """Input and state validation of the ExpIFNeuron."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", math.nan),
            ("v_rest", math.inf),
            ("v_reset", -math.inf),
            ("v_threshold", math.nan),
            ("v_rh", math.inf),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float) -> None:
        """The model rejects non-finite voltage parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            ExpIFNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["delta_t", "tau", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float) -> None:
        """The model rejects non-positive or non-finite scales."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            ExpIFNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["refractory_period", "refractory_remaining"])
    @pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
    def test_rejects_invalid_refractory_values(self, field: str, value: float) -> None:
        """The model rejects invalid refractory values."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            ExpIFNeuron(**kwargs)

    def test_rejects_inconsistent_threshold_relationships(self) -> None:
        """The model rejects inconsistent threshold relationships."""
        with pytest.raises(ValueError, match="must exceed"):
            ExpIFNeuron(v_threshold=-60.0, v_rh=-59.9)
        with pytest.raises(ValueError, match="below v_threshold"):
            ExpIFNeuron(v=30.0)
        with pytest.raises(ValueError, match="below v_threshold"):
            ExpIFNeuron(v_reset=31.0)

    def test_rejects_refractory_remainder_above_period(self) -> None:
        """The model rejects refractory remainder above period."""
        with pytest.raises(ValueError, match="cannot exceed"):
            ExpIFNeuron(refractory_period=0.02, refractory_remaining=0.04)

    @pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float) -> None:
        """A non-finite current is rejected before any state change."""
        neuron = ExpIFNeuron(v=-60.0)
        before = (neuron.v, neuron.refractory_remaining)
        with pytest.raises(ValueError, match="current"):
            neuron.step(current)
        assert (neuron.v, neuron.refractory_remaining) == before

    @pytest.mark.parametrize("runtime_v", [math.nan, 30.0, math.inf])
    def test_rejects_invalid_runtime_voltage_before_update(self, runtime_v: float) -> None:
        """The model rejects invalid runtime voltage before update."""
        neuron = ExpIFNeuron(v=-60.0)
        neuron.v = runtime_v
        with pytest.raises(ValueError, match="runtime voltage state"):
            neuron.step(0.0)

    def test_rejects_invalid_runtime_refractory_state_before_update(self) -> None:
        """The model rejects invalid runtime refractory state before update."""
        neuron = ExpIFNeuron(refractory_period=0.02)
        neuron.refractory_remaining = 0.03
        with pytest.raises(ValueError, match="runtime refractory state"):
            neuron.step(0.0)
        assert neuron.refractory_remaining == 0.03

    def test_rejects_non_finite_rk4_candidate_before_state_mutation(self) -> None:
        """The model rejects non-finite RK4 candidate before state mutation."""
        neuron = ExpIFNeuron(v=-60.0, dt=1.0e308, tau=1.0)
        before = neuron.v
        with pytest.raises(ValueError, match="RK4"):
            neuron.step(1.0e308)
        assert neuron.v == before

    def test_rejects_overflowing_exponential_stage(self) -> None:
        """The model rejects overflowing exponential stage."""
        neuron = ExpIFNeuron(v_threshold=1.0e300, v_rh=0.0, delta_t=1.0)
        with pytest.raises(ValueError, match="exponential term"):
            neuron._rhs(neuron.v_threshold, 0.0)

    def test_rejects_non_finite_derivative(self) -> None:
        """The model rejects non-finite derivative."""
        neuron = ExpIFNeuron(tau=1.0e-308)
        with pytest.raises(ValueError, match="derivative"):
            neuron._rhs(neuron.v, 1.0e308)
