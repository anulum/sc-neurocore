# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHNIsolation from former test_model_fitzhugh_nagumo.py

"""Focused suite: TestFHNIsolation from former test_model_fitzhugh_nagumo.py."""

from __future__ import annotations

from typing import Any, Literal, cast
import math
import numpy as np
import pytest
from sc_neurocore.neurons.models.fitzhugh_nagumo import (
    FitzHughNagumoNeuron,
)


class TestFHNIsolation:
    """The FitzHughNagumoNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = FitzHughNagumoNeuron()
        assert n.v == -1.0 and n.w == -0.5
        assert n.a == 0.7 and n.b == 0.8 and n.epsilon == 0.08
        assert n.integrator == "rk4"

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert FitzHughNagumoNeuron().step(0.0) in (0, 1)

    def test_two_variables_evolve(self) -> None:
        """Both the voltage and the recovery variable evolve."""
        n = FitzHughNagumoNeuron()
        v0, w0 = n.v, n.w
        for _ in range(100):
            n.step(0.5)
        assert n.v != v0 and n.w != w0

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = FitzHughNagumoNeuron()
        for _ in range(100000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = FitzHughNagumoNeuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == -1.0 and n.w == -0.5

    def test_rejects_unknown_integrator(self) -> None:
        """The model rejects unknown integrator."""
        with pytest.raises(ValueError, match="Unsupported integrator"):
            FitzHughNagumoNeuron(integrator=cast(Any, "bad"))

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_cubic_overflow_fails_closed_without_mutating_state(
        self, integrator: Literal["baseline_euler", "rk4", "rosenbrock"]
    ) -> None:
        """Cubic overflow fails closed without mutating state."""
        n = FitzHughNagumoNeuron(v=1e103, w=0.0, integrator=integrator)
        before = (n.v, n.w)

        with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
            n.step(0.0)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": math.nan},
            {"v": True},
            {"w": object()},
            {"w": math.inf},
            {"a": math.nan},
            {"b": 0.0},
            {"epsilon": 0.0},
            {"dt": 0.0},
            {"v_threshold": math.inf},
        ],
    )
    def test_invalid_physical_configuration_is_rejected(self, kwargs: dict[str, Any]) -> None:
        """Invalid physical configuration is rejected."""
        with pytest.raises(ValueError):
            FitzHughNagumoNeuron(**kwargs)

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_runtime_parameter_corruption_fails_before_mutation(
        self, integrator: Literal["baseline_euler", "rk4", "rosenbrock"]
    ) -> None:
        """Runtime parameter corruption fails before mutation."""
        n = FitzHughNagumoNeuron(integrator=integrator)
        n.dt = math.nan
        before = (n.v, n.w)

        with pytest.raises(ValueError):
            n.step(0.5)

        assert (n.v, n.w) == before

    def test_runtime_positive_parameter_corruption_fails_before_mutation(self) -> None:
        """Runtime positive parameter corruption fails before mutation."""
        n = FitzHughNagumoNeuron()
        n.b = 0.0
        before = (n.v, n.w)

        with pytest.raises(ValueError, match="b"):
            n.step(0.5)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_nonfinite_current_fails_before_mutation(
        self, integrator: Literal["baseline_euler", "rk4", "rosenbrock"]
    ) -> None:
        """Non-finite current fails before mutation."""
        n = FitzHughNagumoNeuron(integrator=integrator)
        before = (n.v, n.w)

        with pytest.raises(ValueError, match="current"):
            n.step(math.nan)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize("candidate", [(math.nan, -0.5), (-1.0, math.inf)])
    def test_candidate_validation_rejects_nonfinite_state(
        self, candidate: tuple[float, float]
    ) -> None:
        """Candidate validation rejects non-finite state."""
        n = FitzHughNagumoNeuron()
        with pytest.raises(FloatingPointError, match="candidate"):
            n._validate_candidate(*candidate)

    def test_rhs_rejects_nonfinite_inputs(self) -> None:
        """RHS rejects non-finite inputs."""
        n = FitzHughNagumoNeuron()
        with pytest.raises(FloatingPointError, match="input"):
            n._rhs_tuple(math.inf, n.w, 0.5)

    def test_rhs_rejects_nonfinite_derivative(self) -> None:
        """RHS rejects non-finite derivative."""
        n = FitzHughNagumoNeuron(epsilon=1.0e308)
        with pytest.raises(FloatingPointError, match="derivative"):
            n._rhs_tuple(2.0, -0.5, 0.5)

    def test_rhs_array_adapter_matches_tuple_derivative(self) -> None:
        """RHS array adapter matches tuple derivative."""
        n = FitzHughNagumoNeuron()
        expected = n._rhs_tuple(n.v, n.w, 0.5)
        actual = n._rhs(0.0, np.array([n.v, n.w]), 0.5)
        assert tuple(actual) == pytest.approx(expected)

    def test_rosenbrock_candidate_produces_finite_state(self) -> None:
        """Rosenbrock candidate produces finite state."""
        n = FitzHughNagumoNeuron(integrator="rosenbrock")
        candidate = n._rosenbrock_candidate(0.5)
        assert np.all(np.isfinite(candidate))
