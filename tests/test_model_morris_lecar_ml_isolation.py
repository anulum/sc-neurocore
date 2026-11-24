# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMLIsolation from former test_model_morris_lecar.py

"""Focused suite: TestMLIsolation from former test_model_morris_lecar.py."""

from __future__ import annotations

from typing import Literal
from typing import Any
import math
import numpy as np
import pytest
from sc_neurocore.neurons.models.morris_lecar import (
    MorrisLecarNeuron,
)


class TestMLIsolation:
    """The MorrisLecarNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = MorrisLecarNeuron()
        assert n.v == -60.0 and n.w == 0.0
        assert n.c_m == 20.0 and n.dt == 0.1
        assert n.v_threshold == 0.0
        assert n.integrator == "rk4"

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert MorrisLecarNeuron().step(0.0) in (0, 1)

    def test_both_states_evolve(self) -> None:
        """Both v and w change under drive."""
        n = MorrisLecarNeuron()
        v0, w0 = n.v, n.w
        for _ in range(500):
            n.step(100.0)
        assert n.v != v0 and n.w != w0

    def test_state_finite_long_run(self) -> None:
        """The state stays finite over a long run."""
        n = MorrisLecarNeuron()
        for _ in range(100_000):
            n.step(100.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset_restores_defaults(self) -> None:
        """A reset restores the defaults."""
        n = MorrisLecarNeuron()
        for _ in range(5000):
            n.step(100.0)
        n.reset()
        assert n.v == -60.0 and n.w == 0.0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = MorrisLecarNeuron()
            trace = [(n.step(100.0), n.v, n.w) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_extreme_voltage_rate_overflow_fails_closed(
        self, integrator: Literal["baseline_euler", "rk4", "rosenbrock"]
    ) -> None:
        """Extreme voltage rate overflow fails closed."""
        n = MorrisLecarNeuron(v=1e6, w=0.25, integrator=integrator)
        before = (n.v, n.w)

        with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
            n.step(0.0)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": math.nan},
            {"w": -0.01},
            {"w": 1.01},
            {"c_m": 0.0},
            {"g_ca": 0.0},
            {"g_k": 0.0},
            {"g_l": 0.0},
            {"v2": 0.0},
            {"v4": 0.0},
            {"phi": 0.0},
            {"dt": 0.0},
            {"v_threshold": math.inf},
        ],
    )
    def test_invalid_physical_configuration_is_rejected(self, kwargs: dict[str, Any]) -> None:
        """Invalid physical configuration is rejected."""
        with pytest.raises(ValueError):
            MorrisLecarNeuron(**kwargs)

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_runtime_parameter_corruption_fails_before_mutation(
        self, integrator: Literal["baseline_euler", "rk4", "rosenbrock"]
    ) -> None:
        """Runtime parameter corruption fails before mutation."""
        n = MorrisLecarNeuron(integrator=integrator)
        n.phi = math.nan
        before = (n.v, n.w)

        with pytest.raises(ValueError):
            n.step(100.0)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_potassium_activation_bounds_fail_before_mutation(
        self, integrator: Literal["baseline_euler", "rk4", "rosenbrock"]
    ) -> None:
        """Potassium activation bounds fail before mutation."""
        n = MorrisLecarNeuron(w=1.0, dt=10.0, integrator=integrator)
        before = (n.v, n.w)

        with pytest.raises(FloatingPointError, match="potassium (activation|rate)"):
            n.step(-1_000.0)

        assert (n.v, n.w) == before
