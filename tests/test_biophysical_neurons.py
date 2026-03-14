# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations


class TestConnorStevens:
    def test_fires(self):
        from sc_neurocore.neurons.biophysical import ConnorStevensNeuron

        n = ConnorStevensNeuron()
        spikes = sum(n.step(10.0) for _ in range(100))
        assert spikes > 0

    def test_a_type_current(self):
        from sc_neurocore.neurons.biophysical import ConnorStevensNeuron

        n = ConnorStevensNeuron()
        for _ in range(50):
            n.step(8.0)
        assert n.a != 0.5, "A-type activation must evolve"


class TestWangBuzsaki:
    def test_fires(self):
        from sc_neurocore.neurons.biophysical import WangBuzsakiNeuron

        n = WangBuzsakiNeuron()
        spikes = sum(n.step(1.0) for _ in range(200))
        assert spikes > 0

    def test_fast_spiking(self):
        from sc_neurocore.neurons.biophysical import WangBuzsakiNeuron

        n = WangBuzsakiNeuron()
        spikes = sum(n.step(2.0) for _ in range(200))
        assert spikes >= 3, "fast-spiking interneuron should fire rapidly"


class TestPinskyRinzel:
    def test_fires(self):
        from sc_neurocore.neurons.biophysical import PinskyRinzelNeuron

        n = PinskyRinzelNeuron()
        spikes = sum(n.step(2.0) for _ in range(200))
        assert spikes > 0

    def test_two_compartments(self):
        from sc_neurocore.neurons.biophysical import PinskyRinzelNeuron

        n = PinskyRinzelNeuron()
        for _ in range(100):
            n.step(2.0)
        assert n.v_s != n.v_d, "soma and dendrite must have different voltages"


class TestRulkovMap:
    def test_fires(self):
        from sc_neurocore.neurons.biophysical import RulkovMapNeuron

        n = RulkovMapNeuron(alpha=6.0, sigma=0.0, x_threshold=-0.5)
        spikes = sum(n.step(0.1) for _ in range(2000))
        assert spikes > 0

    def test_deterministic(self):
        from sc_neurocore.neurons.biophysical import RulkovMapNeuron

        n1 = RulkovMapNeuron()
        n2 = RulkovMapNeuron()
        s1 = [n1.step() for _ in range(100)]
        s2 = [n2.step() for _ in range(100)]
        assert s1 == s2


class TestChialvoMap:
    def test_dynamics(self):
        from sc_neurocore.neurons.biophysical import ChialvoMapNeuron

        n = ChialvoMapNeuron()
        for _ in range(100):
            n.step(0.1)
        assert n.x != 0.0 or n.y != 0.0, "state must evolve"


class TestWilsonCowan:
    def test_oscillation(self):
        from sc_neurocore.neurons.biophysical import WilsonCowanUnit

        n = WilsonCowanUnit()
        rates = [n.step(5.0) for _ in range(200)]
        assert max(rates) > min(rates), "should oscillate"


class TestGalvesLocherbach:
    def test_stochastic_firing(self):
        from sc_neurocore.neurons.biophysical import GalvesLocherbachNeuron

        n = GalvesLocherbachNeuron()
        spikes = sum(n.step(2.0) for _ in range(1000))
        assert spikes > 0

    def test_no_fire_without_input(self):
        from sc_neurocore.neurons.biophysical import GalvesLocherbachNeuron

        n = GalvesLocherbachNeuron(steepness=20.0, threshold_rate=5.0)
        spikes = sum(n.step(0.0) for _ in range(100))
        assert spikes == 0


class TestFractionalLIF:
    def test_fires(self):
        from sc_neurocore.neurons.biophysical import FractionalLIFNeuron

        n = FractionalLIFNeuron(alpha=0.9, resistance=5.0)
        spikes = sum(n.step(3.0) for _ in range(200))
        assert spikes > 0

    def test_memory_effect(self):
        from sc_neurocore.neurons.biophysical import FractionalLIFNeuron

        n = FractionalLIFNeuron(alpha=0.5)
        for _ in range(50):
            n.step(0.5)
        assert len(n._history) > 1
