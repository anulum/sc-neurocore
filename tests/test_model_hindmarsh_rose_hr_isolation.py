# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHRIsolation from former test_model_hindmarsh_rose.py

"""Focused suite: TestHRIsolation from former test_model_hindmarsh_rose.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.hindmarsh_rose import (
    HindmarshRoseNeuron,
)


class TestHRIsolation:
    """The HindmarshRoseNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = HindmarshRoseNeuron()
        assert n.x == -1.6 and n.y == -10.0 and n.z == 2.0
        assert n.b == 3.0 and n.r == 0.001 and n.s == 4.0
        assert n.x_rest == -1.6 and n.dt == 0.1
        assert n.integrator == "rk4"

    def test_three_state_variables(self) -> None:
        """The model has three state variables."""
        n = HindmarshRoseNeuron()
        for attr in ["x", "y", "z"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert HindmarshRoseNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self) -> None:
        """The state stays finite over a long run."""
        n = HindmarshRoseNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.x) and np.isfinite(n.y) and np.isfinite(n.z)

    def test_reset_restores_defaults(self) -> None:
        """A reset restores the defaults."""
        n = HindmarshRoseNeuron()
        for _ in range(5000):
            n.step(5.0)
        n.reset()
        assert n.x == -1.6 and n.y == -10.0 and n.z == 2.0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = HindmarshRoseNeuron()
            trace = [(n.step(5.0), n.x) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    def test_rejects_nonfinite_current(self) -> None:
        """The model rejects non-finite current."""
        n = HindmarshRoseNeuron()
        with pytest.raises(ValueError, match="current"):
            n.step(float("nan"))

    def test_rk4_overflow_fails_closed_without_mutating_state(self) -> None:
        """RK4 overflow fails closed without mutating state."""
        n = HindmarshRoseNeuron(x=1e103, y=0.0, z=0.0, integrator="rk4")
        before = (n.x, n.y, n.z)

        with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
            n.step(0.0)

        assert (n.x, n.y, n.z) == before

    def test_euler_overflow_fails_closed_without_mutating_state(self) -> None:
        """Euler overflow fails closed without mutating state."""
        n = HindmarshRoseNeuron(x=1e103, y=0.0, z=0.0, integrator="euler")
        before = (n.x, n.y, n.z)

        with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
            n.step(0.0)

        assert (n.x, n.y, n.z) == before

    def test_runtime_parameter_corruption_fails_before_mutation(self) -> None:
        """Runtime parameter corruption fails before mutation."""
        n = HindmarshRoseNeuron()
        n.dt = float("nan")
        before = (n.x, n.y, n.z)

        with pytest.raises(FloatingPointError, match="non-finite"):
            n.step(3.0)

        assert (n.x, n.y, n.z) == before
