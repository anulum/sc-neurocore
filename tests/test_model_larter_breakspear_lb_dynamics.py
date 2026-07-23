# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLBDynamics from former test_model_larter_breakspear.py

"""Focused suite: TestLBDynamics from former test_model_larter_breakspear.py."""

from __future__ import annotations

from tests.model_larter_breakspear_support import *  # noqa: F403

class TestLBDynamics:
    def test_v_oscillates(self):
        n = LarterBreakspearNeuron()
        vs = [n.step(0.0) for _ in range(10_000)]
        assert np.std(vs) > 0.01

    def test_runge_kutta_tracks_substepped_reference_better_than_euler(self):
        horizon = 0.5
        coupling = 0.15
        reference = LarterBreakspearNeuron(dt=0.0005, integrator="rk4")
        coarse_rk4 = LarterBreakspearNeuron(dt=0.05, integrator="rk4")
        coarse_euler = LarterBreakspearNeuron(dt=0.05, integrator="euler")

        for _ in range(int(horizon / reference.dt)):
            reference.step(coupling)
        for _ in range(int(horizon / coarse_rk4.dt)):
            coarse_rk4.step(coupling)
            coarse_euler.step(coupling)

        rk4_error = abs(coarse_rk4.v - reference.v)
        euler_error = abs(coarse_euler.v - reference.v)

        assert rk4_error < euler_error
        assert rk4_error < 1e-3

    def test_coupling_affects_dynamics(self):
        n1 = LarterBreakspearNeuron()
        n2 = LarterBreakspearNeuron()
        for _ in range(5000):
            n1.step(0.0)
            n2.step(1.0)
        assert n1.v != n2.v

    @pytest.mark.parametrize("coupling", [0.0, 0.5, 1.0, 2.0])
    def test_coupling_sweep(self, coupling: float):
        n = LarterBreakspearNeuron()
        for _ in range(5000):
            n.step(coupling)
        assert np.isfinite(n.v)
