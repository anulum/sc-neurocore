# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPLIFThreshold from former test_model_plif.py

"""Focused suite: TestPLIFThreshold from former test_model_plif.py."""

from __future__ import annotations

from tests.model_plif_support import *  # noqa: F403

class TestPLIFThreshold:
    def test_spike_on_updated_voltage(self):
        """Returned spike is based on updated V, not pre-step V.

        Old V triggers reset (line 33), new V determines returned spike (line 35).
        V_old=1.5 → reset → V_new = 0 + I. Spike returned iff V_new ≥ threshold.
        """
        n = ParametricLIFNeuron()
        n.v = 1.5
        # V_old=1.5 ≥ 1.0 → reset; V_new = alpha*1.5*0 + 0.3 = 0.3 < 1.0
        s = n.step(0.3)
        assert s == 0, "V_new = 0.3 < threshold, should not spike"
        assert abs(n.v - 0.3) < 1e-12

        # Now: V_old=1.5 → reset; V_new = 0 + 1.5 = 1.5 ≥ 1.0 → spike
        n2 = ParametricLIFNeuron()
        n2.v = 1.5
        s2 = n2.step(1.5)
        assert s2 == 1, "V_new = 1.5 ≥ threshold, should spike"

    def test_suprathreshold_input_fires_every_step(self):
        """I ≥ threshold → fires every step (V resets to I ≥ threshold)."""
        n = ParametricLIFNeuron()
        # Skip first step (V starts at 0)
        n.step(1.5)  # V=1.5 (no spike on first since V was 0)
        spikes = sum(n.step(1.5) for _ in range(100))
        assert spikes == 100

    def test_exact_threshold_input(self):
        """I = threshold → fires every step (after first)."""
        n = ParametricLIFNeuron()
        n.step(1.0)  # V=0+1.0=1.0, but spike check was V=0 (no spike)
        # Now V=1.0 → spike, V=0+1.0=1.0
        spikes = sum(n.step(1.0) for _ in range(100))
        assert spikes == 100

    def test_critical_current(self):
        """I_crit = threshold · (1 - alpha). Below this, no spikes ever.

        For alpha=0.5, threshold=1.0: I_crit = 0.5.
        """
        alpha = 0.5
        I_crit = 1.0 * (1.0 - alpha)
        # Just below critical
        n_below = ParametricLIFNeuron(a=0.0)
        spikes_below = sum(n_below.step(I_crit - 0.01) for _ in range(1000))
        assert spikes_below == 0, f"{spikes_below} spikes below I_crit"

    def test_reset_is_soft(self):
        """After spike, V = I (not zero) — soft reset via (1-spike) multiplication."""
        n = ParametricLIFNeuron()
        n.v = 2.0  # will spike
        n.step(0.7)
        assert abs(n.v - 0.7) < 1e-12, "Reset should set V = I, not V = 0"
