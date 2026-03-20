# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Clopath voltage-based STDP

"""Tests for ClopathSTDP (Clopath et al. 2010)."""


from sc_neurocore.synapses.clopath_stdp import ClopathSTDP


class TestClopathSTDP:
    def test_initial_state(self):
        syn = ClopathSTDP()
        assert syn.x_bar == 0.0
        assert syn.u_bar_minus == 0.0
        assert syn.u_bar_plus == 0.0
        assert syn.weight == 0.5

    def test_ltp_with_depolarization(self):
        """Pre spike during strong depolarization → LTP."""
        syn = ClopathSTDP(a_ltp=0.01, weight=0.3)
        # Warm up voltage traces above thresholds
        for _ in range(50):
            syn.step(pre_spike=False, u_post=-30.0, dt=0.5)
        # Now fire with high voltage
        for _ in range(50):
            syn.step(pre_spike=True, u_post=-20.0, dt=0.5)
        assert syn.weight > 0.3

    def test_ltd_with_pre_spike_above_rest(self):
        """Pre spike with post above rest → LTD."""
        syn = ClopathSTDP(a_ltd=0.01, a_ltp=0.0, weight=0.7)
        # Build up x_bar, then fire pre spikes when post is slightly depolarized
        for _ in range(100):
            syn.step(pre_spike=True, u_post=-50.0, dt=1.0)
        assert syn.weight < 0.7

    def test_no_change_at_rest(self):
        """At resting potential with no spikes → no weight change."""
        syn = ClopathSTDP(weight=0.5)
        w_before = syn.weight
        for _ in range(100):
            syn.step(pre_spike=False, u_post=-70.6, dt=1.0)
        assert syn.weight == w_before

    def test_weight_bounds(self):
        """Weight should stay in [w_min, w_max]."""
        syn = ClopathSTDP(a_ltp=1.0, w_min=0.0, w_max=1.0, weight=0.9)
        for _ in range(200):
            syn.step(pre_spike=True, u_post=-10.0, dt=1.0)
        assert 0.0 <= syn.weight <= 1.0

    def test_reset(self):
        syn = ClopathSTDP()
        syn.step(pre_spike=True, u_post=-40.0, dt=1.0)
        assert syn.x_bar > 0
        syn.reset()
        assert syn.x_bar == 0.0
        assert syn.u_bar_minus == 0.0
        assert syn.u_bar_plus == 0.0

    def test_trace_decay(self):
        """Pre trace should decay exponentially without new spikes."""
        syn = ClopathSTDP(tau_x=10.0)
        syn.step(pre_spike=True, u_post=-70.0, dt=1.0)
        x_after_spike = syn.x_bar
        syn.step(pre_spike=False, u_post=-70.0, dt=10.0)
        assert syn.x_bar < x_after_spike * 0.5
