# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for triplet STDP

"""Tests for triplet STDP (Pfister-Gerstner 2006)."""

from sc_neurocore.synapses.triplet_stdp import TripletSTDP


class TestTripletSTDP:
    def test_pre_then_post_potentiates(self):
        s = TripletSTDP(weight=0.5)
        # Pre spike at t=0
        s.step(pre_spike=True, post_spike=False, dt=1.0)
        # Post spike at t=5ms — should potentiate
        for _ in range(4):
            s.step(pre_spike=False, post_spike=False, dt=1.0)
        s.step(pre_spike=False, post_spike=True, dt=1.0)
        assert s.weight > 0.5

    def test_post_then_pre_depresses(self):
        s = TripletSTDP(weight=0.5)
        # Post spike at t=0
        s.step(pre_spike=False, post_spike=True, dt=1.0)
        # Pre spike at t=5ms — should depress
        for _ in range(4):
            s.step(pre_spike=False, post_spike=False, dt=1.0)
        s.step(pre_spike=True, post_spike=False, dt=1.0)
        assert s.weight < 0.5

    def test_triplet_ltp_stronger_than_pair(self):
        """With recent post history (o2 > 0), LTP should be larger."""
        # Pair-only
        s1 = TripletSTDP(weight=0.5, a3_plus=0.0)
        s1.step(pre_spike=True, post_spike=False, dt=1.0)
        s1.step(pre_spike=False, post_spike=True, dt=1.0)
        pair_ltp = s1.weight - 0.5

        # Triplet (with prior post spike creating o2)
        s2 = TripletSTDP(weight=0.5)
        s2.step(pre_spike=False, post_spike=True, dt=1.0)  # creates o2
        for _ in range(5):
            s2.step(pre_spike=False, post_spike=False, dt=1.0)
        s2.step(pre_spike=True, post_spike=False, dt=1.0)
        s2.step(pre_spike=False, post_spike=True, dt=1.0)
        triplet_ltp = s2.weight - 0.5

        assert triplet_ltp > pair_ltp

    def test_weight_bounds(self):
        s = TripletSTDP(weight=0.99, w_max=1.0)
        for _ in range(100):
            s.step(pre_spike=True, post_spike=True, dt=1.0)
        assert s.weight <= 1.0

        s2 = TripletSTDP(weight=0.01, w_min=0.0)
        for _ in range(100):
            # Post then pre → depression
            s2.step(pre_spike=False, post_spike=True, dt=1.0)
            s2.step(pre_spike=True, post_spike=False, dt=1.0)
        assert s2.weight >= 0.0

    def test_reset(self):
        s = TripletSTDP(weight=0.5)
        s.step(pre_spike=True, post_spike=True, dt=1.0)
        s.reset()
        assert s.r1 == 0.0
        assert s.o1 == 0.0
        assert s.r2 == 0.0
        assert s.o2 == 0.0

    def test_no_spikes_no_change(self):
        s = TripletSTDP(weight=0.5)
        for _ in range(100):
            s.step(pre_spike=False, post_spike=False, dt=1.0)
        assert s.weight == 0.5
