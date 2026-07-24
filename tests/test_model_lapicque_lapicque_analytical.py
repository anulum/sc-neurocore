# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicqueAnalytical from former test_model_lapicque.py

"""Focused suite: TestLapicqueAnalytical from former test_model_lapicque.py."""

from __future__ import annotations

from tests.model_lapicque_support import *  # noqa: F403


class TestLapicqueAnalytical:
    def test_exact_flow_formula(self):
        """V_next = V_ss + (V - V_ss) · exp(-dt / τ)."""
        n = LapicqueNeuron()
        v0 = n.v
        I = 0.5  # subthreshold
        v_ss = n.v_rest + n.resistance * I
        expected = v_ss + (v0 - v_ss) * np.exp(-n.dt / n.tau)
        n.step(I)
        assert abs(n.v - expected) < 1e-14

    def test_exact_flow_separates_from_forward_euler_for_large_dt(self):
        n = LapicqueNeuron(v=0.25, dt=5.0)
        v0 = n.v
        current = 0.5
        euler = v0 + (-(v0 - n.v_rest) + n.resistance * current) / n.tau * n.dt
        v_ss = n.v_rest + n.resistance * current
        expected = v_ss + (v0 - v_ss) * np.exp(-n.dt / n.tau)
        spike = n.step(current)
        assert spike == 0
        assert abs(n.v - expected) < 1e-14
        assert abs(n.v - euler) > 1e-4

    def test_steady_state(self):
        """V_ss = V_rest + R·I (at equilibrium dV=0)."""
        n = LapicqueNeuron()
        I = 0.5  # subthreshold
        for _ in range(10_000):
            n.step(I)
        expected_ss = n.v_rest + n.resistance * I
        assert abs(n.v - expected_ss) < 0.01

    def test_rheobase(self):
        """Rheobase = (V_threshold - V_rest) / R. Below: silent."""
        n = LapicqueNeuron()
        rheobase = (n.v_threshold - n.v_rest) / n.resistance
        # Below rheobase: no spikes
        assert len(_run(n, current=rheobase * 0.9, steps=5000)) == 0

    def test_above_rheobase_fires(self):
        n = LapicqueNeuron()
        rheobase = (n.v_threshold - n.v_rest) / n.resistance
        assert len(_run(n, current=rheobase * 1.5, steps=5000)) >= 10

    def test_spike_resets_voltage(self):
        n = LapicqueNeuron()
        for _ in range(10_000):
            if n.step(20.0) == 1:
                assert n.v == n.v_reset
                break

    def test_resistance_scales_input(self):
        """Higher R → more effective current."""
        n1 = LapicqueNeuron(resistance=0.5, v_threshold=100.0)
        n2 = LapicqueNeuron(resistance=2.0, v_threshold=100.0)
        for _ in range(100):
            n1.step(10.0)
            n2.step(10.0)
        assert n2.v > n1.v
