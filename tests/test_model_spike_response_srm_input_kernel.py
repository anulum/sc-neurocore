# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRMInputKernel from former test_model_spike_response.py

"""Focused suite: TestSRMInputKernel from former test_model_spike_response.py."""

from __future__ import annotations

from tests.model_spike_response_support import *  # noqa: F403


class TestSRMInputKernel:
    def test_kappa_formula_exact(self):
        """κ = I·(1 - exp(-dt/tau_kappa))."""
        n = SpikeResponseNeuron()
        n.time_since_spike = 1000.0
        n.step(5.0)
        expected = _kappa(5.0, n.dt, n.tau_kappa)
        assert abs(n.v - expected) < 1e-10

    def test_kappa_linear_in_I(self):
        """κ(2I) = 2·κ(I) — linearity."""
        k3 = _kappa(3.0, 1.0, 5.0)
        k6 = _kappa(6.0, 1.0, 5.0)
        assert abs(k6 - 2 * k3) < 1e-10

    def test_kappa_decreases_with_tau_kappa(self):
        """Larger tau_kappa → smaller κ (slower integration)."""
        k_small_tau = _kappa(10.0, 1.0, 1.0)
        k_large_tau = _kappa(10.0, 1.0, 20.0)
        assert k_small_tau > k_large_tau

    def test_critical_current(self):
        """I_crit = θ / (1 - exp(-dt/tau_kappa)). Verified above/below."""
        n = SpikeResponseNeuron()
        I_crit = n.v_threshold / (1.0 - np.exp(-n.dt / n.tau_kappa))
        n_below = SpikeResponseNeuron()
        n_below.time_since_spike = 1000.0
        assert n_below.step(I_crit * 0.9) == 0
        n_above = SpikeResponseNeuron()
        n_above.time_since_spike = 1000.0
        assert n_above.step(I_crit * 1.1) == 1
