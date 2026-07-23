# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorIsolation from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorIsolation from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403

class TestPerfectIntegratorIsolation:
    """Core single-neuron dynamics."""

    def test_construction_defaults(self):
        n = PerfectIntegratorNeuron()
        assert n.v == 0.0
        assert n.c_m == 1.0
        assert n.v_threshold == 1.0
        assert n.v_reset == 0.0
        assert n.dt == 0.1

    def test_step_returns_binary(self):
        assert PerfectIntegratorNeuron().step(0.0) in (0, 1)

    def test_zero_input_no_drift(self):
        """With I=0, voltage must stay exactly at initial value."""
        n = PerfectIntegratorNeuron(v=0.3)
        for _ in range(1000):
            n.step(0.0)
        assert n.v == 0.3

    def test_linear_voltage_ramp(self):
        """Voltage should increase linearly: V(t) = V₀ + (I/C)·dt·t."""
        n = PerfectIntegratorNeuron()
        I, C, dt = 3.0, n.c_m, n.dt
        dv = I / C * dt  # 0.3 per step
        for t in range(1, 4):
            n.step(I)
            expected = dv * t
            assert abs(n.v - expected) < 1e-12, f"step {t}: {n.v} != {expected}"

    def test_no_leak_invariant(self):
        """Key property: voltage is unchanged by zero-input steps (no decay)."""
        n = PerfectIntegratorNeuron()
        n.step(2.0)
        v_charged = n.v
        for _ in range(500):
            n.step(0.0)
        assert n.v == v_charged, "Voltage decayed — leak detected in integrator"
