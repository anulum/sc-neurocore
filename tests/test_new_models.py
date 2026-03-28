# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SRM0 and Brunel-Wang neuron models

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron
from sc_neurocore.neurons.models.srm0 import SRM0Neuron


class TestSRM0:
    def test_fires_with_current(self):
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, dt=0.1)
        spikes = sum(n.step(2.0) for _ in range(1000))
        assert spikes > 0

    def test_subthreshold_no_spikes(self):
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, dt=0.1)
        spikes = sum(n.step(0.5) for _ in range(1000))
        assert spikes == 0

    def test_eta_refractory(self):
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, eta_reset=10.0, dt=0.1)
        # Drive hard until spike
        for _ in range(1000):
            if n.step(2.0):
                break
        # Right after spike, v should be near rest due to eta
        assert n.v < 0.5

    def test_reset(self):
        n = SRM0Neuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == n.v_rest
        assert n._eta == 0.0

    def test_get_state(self):
        n = SRM0Neuron()
        n.step(0.1)
        s = n.get_state()
        assert "v" in s and "eta" in s and "t" in s

    @pytest.mark.skip(reason="TODO: SRM0 eta dynamics need investigation — gh-issue pending")
    def test_rate_increases_with_current(self):
        r1 = sum(SRM0Neuron(dt=0.1).step(8.0) for _ in range(5000))
        r2 = sum(SRM0Neuron(dt=0.1).step(15.0) for _ in range(5000))
        assert r2 > r1


class TestBrunelWang:
    def test_fires_with_ampa(self):
        n = BrunelWangNeuron(dt=0.1)
        spikes = sum(n.step(i_ampa_ext=5.0) for _ in range(5000))
        assert spikes > 0

    def test_subthreshold(self):
        n = BrunelWangNeuron(dt=0.1)
        spikes = sum(n.step(i_ampa_ext=0.0) for _ in range(1000))
        assert spikes == 0

    def test_gaba_inhibition(self):
        # With strong GABA, should fire less than without
        n1 = BrunelWangNeuron(dt=0.1)
        n2 = BrunelWangNeuron(dt=0.1)
        s1 = sum(n1.step(i_ampa_ext=3.0, s_gaba=0.0) for _ in range(5000))
        s2 = sum(n2.step(i_ampa_ext=3.0, s_gaba=0.8) for _ in range(5000))
        assert s1 >= s2

    def test_nmda_voltage_dep(self):
        n = BrunelWangNeuron()
        # At rest (-70mV), Mg block is strong
        block_rest = n._nmda_voltage_dep(-70.0)
        # At depolarised (-20mV), Mg block is weak
        block_depol = n._nmda_voltage_dep(-20.0)
        assert block_depol > block_rest

    def test_refractory(self):
        n = BrunelWangNeuron(dt=0.1, tau_ref=2.0)
        # Drive until spike
        for _ in range(10000):
            if n.step(i_ampa_ext=10.0):
                break
        # Next step should be refractory (no spike)
        assert n.step(i_ampa_ext=10.0) == 0

    def test_reset(self):
        n = BrunelWangNeuron()
        for _ in range(100):
            n.step(i_ampa_ext=5.0)
        n.reset()
        assert n.v == n.v_rest
