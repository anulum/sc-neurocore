# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNakerLIFDynamics from former test_model_spinnaker_lif.py

"""Focused suite: TestSpiNNakerLIFDynamics from former test_model_spinnaker_lif.py."""

from __future__ import annotations

from tests.model_spinnaker_lif_support import *  # noqa: F403

class TestSpiNNakerLIFDynamics:
    def test_membrane_equation(self):
        """Exact LIF flow solves constant-current membrane dynamics."""
        n = SpiNNakerLIFNeuron(tau_refrac=0.0)
        v0 = n.v
        current = 15.0
        n.step(current)
        steady = n.v_rest + current + n.i_offset
        expected = steady + (v0 - steady) * math.exp(-n.dt / n.tau_m)
        assert abs(n.v - expected) < 1e-10

    def test_exact_flow_reduces_to_euler_order_for_small_dt(self):
        n = SpiNNakerLIFNeuron(dt=1.0e-6, tau_refrac=0.0)
        v0 = n.v
        current = 15.0
        n.step(current)
        euler = v0 + (-(v0 - n.v_rest) + current + n.i_offset) / n.tau_m * n.dt
        assert abs(n.v - euler) < 1e-12

    def test_steady_state(self):
        """V_ss = V_rest + I. At I=10: V_ss = -60, below threshold."""
        n = SpiNNakerLIFNeuron()
        for _ in range(10000):
            n.step(10.0)
        assert abs(n.v - (-60.0)) < 0.1

    def test_monotonic_fi(self):
        rates = []
        for I in [25.0, 30.0, 40.0, 50.0]:
            n = SpiNNakerLIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
