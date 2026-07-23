# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTMSTPDynamics from former test_model_tsodyks_markram.py

"""Focused suite: TestTMSTPDynamics from former test_model_tsodyks_markram.py."""

from __future__ import annotations

from tests.model_tsodyks_markram_support import *  # noqa: F403

class TestTMSTPDynamics:
    """Short-term plasticity: x (depression) and u (facilitation)."""

    def test_x_depletes_on_presyn_spike(self):
        """x decreases on presynaptic spike (depression)."""
        n = TsodyksMarkramNeuron()
        x0 = n.x
        n.step(0.0, presynaptic_spike=True)
        assert n.x < x0, f"x={n.x}, expected depletion"

    def test_x_recovers_between_spikes(self):
        """x recovers toward 1.0 with tau_d between spikes."""
        n = TsodyksMarkramNeuron()
        n.step(0.0, presynaptic_spike=True)  # deplete
        x_after_spike = n.x
        for _ in range(2000):
            n.step(0.0)  # no presyn spike, x recovers
        assert n.x > x_after_spike

    def test_u_facilitates_on_presyn_spike(self):
        """u increases on presynaptic spike (facilitation)."""
        n = TsodyksMarkramNeuron()
        u0 = n.u
        n.step(0.0, presynaptic_spike=True)
        assert n.u > u0

    def test_u_decays_between_spikes(self):
        """u decays toward u_se with tau_f between spikes."""
        n = TsodyksMarkramNeuron()
        n.step(0.0, presynaptic_spike=True)  # facilitate
        u_after = n.u
        for _ in range(5000):
            n.step(0.0)
        assert abs(n.u - n.u_se) < abs(u_after - n.u_se)

    def test_depression_reduces_efficacy(self):
        """Repeated presyn spikes deplete x → weaker synaptic current."""
        n = TsodyksMarkramNeuron()
        # First spike: high x
        n.step(0.0, presynaptic_spike=True)
        x1 = n.x
        # Second spike: lower x
        n.step(0.0, presynaptic_spike=True)
        x2 = n.x
        assert x2 < x1, "x should deplete further on second spike"

    def test_x_bounded_0_1(self):
        """x stays in [0, 1]."""
        n = TsodyksMarkramNeuron()
        for _ in range(1000):
            n.step(0.0, presynaptic_spike=(np.random.random() < 0.5))
        assert 0.0 <= n.x <= 1.0

    def test_u_bounded_0_1(self):
        """u stays in [0, 1]."""
        n = TsodyksMarkramNeuron()
        for _ in range(1000):
            n.step(0.0, presynaptic_spike=(np.random.random() < 0.5))
        assert 0.0 <= n.u <= 1.0
