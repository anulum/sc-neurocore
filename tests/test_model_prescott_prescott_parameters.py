# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrescottParameters from former test_model_prescott.py

"""Focused suite: TestPrescottParameters from former test_model_prescott.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.prescott import (
    PrescottNeuron,
)
from tests.model_prescott_support import (
    _run,
)


class TestPrescottParameters:
    """Parameter behaviour of the PrescottNeuron."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("w", np.inf),
            ("g_fast", -1.0),
            ("g_slow", -1.0),
            ("g_l", -1.0),
            ("gamma_w", 0.0),
            ("tau_w", 0.0),
            ("phi", -0.1),
            ("dt", 0.0),
            ("v_threshold", np.inf),
        ],
    )
    def test_rejects_invalid_configuration(self, field: str, value: float) -> None:
        """The model rejects invalid configuration."""
        with pytest.raises((ValueError, FloatingPointError)):
            kwargs: dict[str, Any] = {field: value}
            PrescottNeuron(**kwargs)

    def test_rejects_non_finite_current_before_state_mutation(self) -> None:
        """A non-finite current is rejected before any state change."""
        n = PrescottNeuron()
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="current"):
            n.step(float("nan"))
        assert (n.v, n.w) == before

    def test_rejects_corrupted_state_before_mutation(self) -> None:
        """The model rejects corrupted state before mutation."""
        n = PrescottNeuron()
        n.w = 1.5
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="w state"):
            n.step(50.0)
        assert (n.v, n.w) == before

    def test_state_kernel_rejects_non_finite_voltage(self) -> None:
        """State kernel rejects non-finite voltage."""
        with pytest.raises(FloatingPointError, match="voltage state"):
            PrescottNeuron._validate_state(float("nan"), 0.0)

    def test_recovery_kernel_rejects_non_finite_state(self) -> None:
        """Recovery kernel rejects non-finite state."""
        with pytest.raises(FloatingPointError, match="w state"):
            PrescottNeuron._validate_recovery(float("inf"))

    def test_rejects_non_finite_derivative_before_state_mutation(self) -> None:
        """The model rejects non-finite derivative before state mutation."""
        n = PrescottNeuron(g_fast=1.0e308)
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="derivative"):
            n.step(50.0)
        assert (n.v, n.w) == before

    def test_g_slow_affects_dynamics(self) -> None:
        """Different g_slow values produce different spike patterns.

        The slow K conductance interacts non-linearly with the fast
        subsystem — relationship is not simply monotonic.
        """
        n1 = PrescottNeuron(g_slow=10.0)
        n2 = PrescottNeuron(g_slow=30.0)
        s1 = len(_run(n1, current=50.0, steps=100000))
        s2 = len(_run(n2, current=50.0, steps=100000))
        assert s1 != s2, "g_slow had no effect on dynamics"

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = PrescottNeuron(dt=dt)
        for _ in range(50000):
            n.step(50.0)
        assert np.isfinite(n.v)

    def test_upward_crossing_only(self) -> None:
        """Spikes only on V upward crossing of threshold."""
        n = PrescottNeuron()
        prev_v = n.v
        crossings = 0
        for _ in range(50000):
            s = n.step(50.0)
            if s == 1:
                crossings += 1
                assert prev_v < n.v_threshold
                assert n.v >= n.v_threshold
            prev_v = n.v
        assert crossings > 0
