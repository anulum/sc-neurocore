# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRMRefractoryKernel from former test_model_spike_response.py

"""Focused suite: TestSRMRefractoryKernel from former test_model_spike_response.py."""

from __future__ import annotations

from tests.model_spike_response_support import *  # noqa: F403

class TestSRMRefractoryKernel:
    def test_eta_at_tss_zero(self):
        """After spike, tss=0. Next step: η uses tss=0 → η = eta_reset exactly."""
        n = SpikeResponseNeuron()
        n.step(10.0)  # spike, tss → 0
        assert n.time_since_spike == 0.0
        # Next step: eta(tss=0) = eta_reset = -5.0
        n.step(0.0)  # zero input → v = eta(0) + 0 = -5.0
        assert abs(n.v - n.eta_reset) < 1e-10, f"v={n.v}, expected eta_reset={n.eta_reset}"

    def test_eta_decays_step_by_step(self):
        """Track η decay: at step k after spike, η uses tss = k-1."""
        n = SpikeResponseNeuron()
        n.step(10.0)  # spike → tss = 0
        for k in range(1, 15):
            n.step(0.0)
            # eta was computed at tss = k-1 (before increment)
            expected_eta = _eta(float(k - 1), n.eta_reset, n.tau_eta)
            assert abs(n.v - expected_eta) < 1e-6, (
                f"Step {k} after spike: v={n.v:.6f}, eta(tss={k - 1})={expected_eta:.6f}"
            )

    def test_eta_zero_beyond_100(self):
        """η clipped to 0 when tss ≥ 100 (code optimisation)."""
        n = SpikeResponseNeuron()
        n.time_since_spike = 100.0
        n.step(0.0)
        assert n.v == 0.0

    def test_refractory_prevents_immediate_respike(self):
        """Strong η suppression prevents re-spike even with strong input."""
        n = SpikeResponseNeuron()
        n.step(10.0)  # spike
        s = n.step(10.0)  # eta(0) = -5, kappa = 1.81 → v ≈ -3.19
        assert s == 0

    def test_v_after_spike_equals_eta_plus_kappa(self):
        """Verify v = η(tss) + κ(I) exactly for several steps post-spike."""
        n = SpikeResponseNeuron()
        n.step(10.0)  # spike
        I = 8.0
        for k in range(1, 10):
            n.step(I)
            expected = _eta(float(k - 1), n.eta_reset, n.tau_eta) + _kappa(I, n.dt, n.tau_kappa)
            assert abs(n.v - expected) < 1e-6, f"k={k}: v={n.v:.6f}, expected={expected:.6f}"
