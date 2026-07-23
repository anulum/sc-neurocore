# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRM0Dynamics from former test_model_srm0.py

"""Focused suite: TestSRM0Dynamics from former test_model_srm0.py."""

from __future__ import annotations

from tests.model_srm0_support import *  # noqa: F403

class TestSRM0Dynamics:
    def test_voltage_integrates(self) -> None:
        """Unlike SpikeResponseNeuron, SRM0 accumulates V over steps."""
        n = SRM0Neuron()
        v_prev = n.v
        for _ in range(5):
            n.step(0.5)
        assert n.v > v_prev  # V grew from integration

    def test_exact_flow_one_step(self) -> None:
        n = SRM0Neuron()
        n._eta = -2.0
        expected_v, expected_eta = _exact_reference(n, current=0.5)
        n.step(0.5)
        assert abs(n.v - expected_v) < 1e-12
        assert abs(n._eta - expected_eta) < 1e-12

    def test_exact_flow_differs_from_membrane_euler(self) -> None:
        n = SRM0Neuron()
        n._eta = -2.0
        current = 0.5
        eta_euler = n._eta * math.exp(-n.dt / n.tau_eta)
        euler_v = n.v + (n.resistance * current - (n.v - (n.v_rest + eta_euler))) * n.dt / n.tau_m
        n.step(current)
        assert abs(n.v - euler_v) > 1e-5

    def test_invalid_runtime_current_preserves_state(self) -> None:
        n = SRM0Neuron()
        before = (n.v, n._eta, n._t, n._last_spike_time)
        with pytest.raises(ValueError, match="current"):
            n.step(float("nan"))
        assert (n.v, n._eta, n._t, n._last_spike_time) == before

    def test_steady_state_subthreshold(self) -> None:
        """V_ss = R·I when eta=0 (no recent spike)."""
        n = SRM0Neuron()
        for _ in range(10000):
            n.step(0.5)
        v_ss = n.resistance * 0.5
        assert abs(n.v - v_ss) < 0.01

    def test_refractory_lengthens_isi(self) -> None:
        """eta_reset > 0 → longer ISI than pure LIF."""
        n_refrac = SRM0Neuron(eta_reset=5.0)
        n_norefrac = SRM0Neuron(eta_reset=0.0)
        s_refrac = len(_run(n_refrac, current=5.0, steps=10000))
        s_norefrac = len(_run(n_norefrac, current=5.0, steps=10000))
        assert s_norefrac > s_refrac
