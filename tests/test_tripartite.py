# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for tripartite synapse

"""Tests for TripartiteSynapse (astrocyte ↔ synapse coupling)."""

from sc_neurocore.synapses.tripartite import TripartiteSynapse


class TestTripartiteSynapse:
    def test_initial_state(self):
        syn = TripartiteSynapse()
        assert syn.weight == 0.5
        assert syn.ca > 0
        assert syn.ip3 > 0

    def test_pre_spike_drives_ip3(self):
        """Pre-synaptic spikes should increase astrocyte IP3."""
        syn = TripartiteSynapse()
        ip3_before = syn.ip3
        for _ in range(100):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        assert syn.ip3 > ip3_before

    def test_sustained_activity_raises_ip3(self):
        """Sustained pre-synaptic activity should raise astrocyte IP3."""
        syn = TripartiteSynapse(glut_per_spike=5.0)
        ip3_start = syn.ip3
        for _ in range(500):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        assert syn.ip3 > ip3_start

    def test_facilitation_increases_weight(self):
        """When astrocyte Ca exceeds threshold, weight should increase."""
        syn = TripartiteSynapse(
            base_weight=0.3,
            glut_per_spike=10.0,
            ca_threshold=0.005,
            facilitation=5.0,
            w_max=1.0,
        )
        for _ in range(1000):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        # With low ca_threshold, even the residual Ca activity triggers facilitation
        assert syn.weight > 0.3

    def test_no_activity_returns_to_baseline(self):
        """Without pre-synaptic activity, weight drifts toward baseline."""
        syn = TripartiteSynapse(base_weight=0.5, depression_rate=0.1)
        syn.weight = 0.8
        for _ in range(200):
            syn.step(pre_spike=False, post_spike=False, dt=0.01)
        assert syn.weight < 0.8

    def test_weight_bounds(self):
        """Weight should stay in [w_min, w_max]."""
        syn = TripartiteSynapse(
            facilitation=100.0,
            glut_per_spike=50.0,
            ca_threshold=0.01,
            w_min=0.0,
            w_max=1.0,
        )
        for _ in range(2000):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        assert 0.0 <= syn.weight <= 1.0

    def test_effective_weight(self):
        syn = TripartiteSynapse(base_weight=0.4)
        assert syn.effective_weight() == 0.4

    def test_reset(self):
        syn = TripartiteSynapse(base_weight=0.5)
        for _ in range(100):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        syn.reset()
        assert syn.weight == 0.5
        assert syn.astrocyte.ca == 0.05
        assert syn._glut_current == 0.0
