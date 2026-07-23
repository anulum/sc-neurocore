# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestILIFAnalytical from former test_model_ilif.py

"""Focused suite: TestILIFAnalytical from former test_model_ilif.py."""

from __future__ import annotations

from tests.model_ilif_support import *  # noqa: F403

class TestILIFAnalytical:
    def test_v_update_formula(self):
        """V = alpha_m·V + I - inh_strength·inh_trace (after trace decay)."""
        n = InhibitoryLIFNeuron()
        v0, inh0 = n.v, n.inh_trace
        I = 0.5
        # Trace decays first
        inh_after = inh0 * n.alpha_inh
        expected_v = n.alpha_m * v0 + I - n.inh_strength * inh_after
        n.step(I)
        if n.v != n.v_reset:  # no spike
            assert abs(n.v - expected_v) < 1e-12

    def test_inh_trace_decay(self):
        """inh_trace *= alpha_inh per step."""
        n = InhibitoryLIFNeuron()
        n.inh_trace = 1.0
        steps = 10
        for _ in range(steps):
            n.step(0.0)
        expected = 1.0 * n.alpha_inh**steps
        assert abs(n.inh_trace - expected) < 1e-10

    def test_spike_increments_trace(self):
        """On spike: inh_trace += 1."""
        n = InhibitoryLIFNeuron()
        for _ in range(10_000):
            inh_before = n.inh_trace
            if n.step(5.0) == 1:
                # Trace was decayed, then incremented by 1
                expected = inh_before * n.alpha_inh + 1.0
                assert abs(n.inh_trace - expected) < 1e-10
                break

    def test_spike_resets_voltage(self):
        n = InhibitoryLIFNeuron()
        for _ in range(10_000):
            if n.step(5.0) == 1:
                assert n.v == n.v_reset
                break

    def test_inhibition_suppresses_after_spike(self):
        """After spike, inh_trace > 0 → suppresses next V integration."""
        n = InhibitoryLIFNeuron()
        for _ in range(10_000):
            if n.step(5.0) == 1:
                assert n.inh_trace > 0
                # Next step: V = alpha_m·0 + I - strength·trace < I
                v_next_no_inh = 5.0  # just current
                n.step(5.0)
                assert n.v < v_next_no_inh
                break

    def test_alpha_m_range(self):
        """0 < alpha_m < 1 for finite tau_m."""
        n = InhibitoryLIFNeuron()
        assert 0 < n.alpha_m < 1
