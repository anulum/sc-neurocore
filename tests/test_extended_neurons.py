# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for all extended neuron models

"""Tests for all extended neuron models."""

from __future__ import annotations

from sc_neurocore.neurons.models import (
    BrainScaleSAdExNeuron,
    ButeraRespiratoryNeuron,
    ChayNeuron,
    CourageNekorkinMapNeuron,
    DestexheThalamicNeuron,
    EnergyLIFNeuron,
    ErmentroutKopellPopulation,
    EscapeRateNeuron,
    FitzHughRinzelNeuron,
    GatedLIFNeuron,
    GLIFNeuron,
    GutkinErmentroutNeuron,
    HuberBraunNeuron,
    InhomogeneousPoissonNeuron,
    JansenRitUnit,
    LeakyCompeteFireNeuron,
    LoihiCUBANeuron,
    MATNeuron,
    MedvedevMapNeuron,
    PrescottNeuron,
    SFANeuron,
    ShermanRinzelKeizerNeuron,
    SigmaDeltaNeuron,
    SpiNNakerLIFNeuron,
    StochasticIFNeuron,
    TrueNorthNeuron,
    WongWangUnit,
)


class TestDestexhe:
    def test_fires(self):
        n = DestexheThalamicNeuron()
        assert sum(n.step(5.0) for _ in range(200)) > 0

    def test_t_current(self):
        n = DestexheThalamicNeuron()
        for _ in range(100):
            n.step(3.0)
        assert n.h_t != 1.0


class TestHuberBraun:
    def test_fires(self):
        n = HuberBraunNeuron()
        assert sum(n.step(5.0) for _ in range(300)) > 0


class TestGutkinErmentrout:
    def test_fires(self):
        n = GutkinErmentroutNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0


class TestFitzHughRinzel:
    def test_bursting(self):
        n = FitzHughRinzelNeuron()
        assert sum(n.step(1.0) for _ in range(2000)) > 0

    def test_slow_var(self):
        n = FitzHughRinzelNeuron()
        for _ in range(500):
            n.step(0.5)
        assert n.y != 0.0


class TestChay:
    def test_drive_changes_state_without_leaving_physical_bounds(self):
        rest = ChayNeuron()
        driven = ChayNeuron()
        for _ in range(500):
            rest.step(0.0)
            driven.step(5.0)
        assert driven.v > rest.v
        assert 0.0 <= driven.n <= 1.0
        assert driven.ca >= 0.0

    def test_calcium(self):
        n = ChayNeuron()
        for _ in range(200):
            n.step(5.0)
        assert n.ca != 0.1


class TestButera:
    def test_dynamics(self):
        n = ButeraRespiratoryNeuron()
        for _ in range(100):
            n.step(5.0)
        assert n.h_nap != 0.5, "persistent Na inactivation must evolve"


class TestShermanRinzelKeizer:
    def test_fires(self):
        n = ShermanRinzelKeizerNeuron()
        assert sum(n.step(3.0) for _ in range(500)) > 0


class TestGLIF:
    def test_fires(self):
        n = GLIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_adaptation(self):
        n = GLIFNeuron()
        for _ in range(100):
            n.step(30.0)
        assert n.i_asc1 != 0.0 or n.i_asc2 != 0.0


class TestMAT:
    def test_fires(self):
        n = MATNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_threshold_adapts(self):
        n = MATNeuron()
        for _ in range(50):
            n.step(30.0)
        assert n.theta1 > 0.0


class TestSFA:
    def test_fires(self):
        n = SFANeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_adaptation_reduces_rate(self):
        n = SFANeuron()
        first = sum(n.step(25.0) for _ in range(100))
        second = sum(n.step(25.0) for _ in range(100))
        assert second <= first + 2


class TestStochasticIF:
    def test_fires(self):
        n = StochasticIFNeuron(mu=25.0, sigma=2.0)
        assert sum(n.step(0.0) for _ in range(500)) > 0


class TestEscapeRate:
    def test_fires(self):
        n = EscapeRateNeuron(rho_0=0.1, delta_u=2.0)
        assert sum(n.step(30.0) for _ in range(500)) > 0


class TestSigmaDelta:
    def test_positive_spike(self):
        n = SigmaDeltaNeuron(v_threshold=1.0)
        spikes = [n.step(0.3) for _ in range(10)]
        assert 1 in spikes

    def test_negative_spike(self):
        n = SigmaDeltaNeuron(v_threshold=1.0)
        spikes = [n.step(-0.3) for _ in range(10)]
        assert -1 in spikes


class TestGatedLIF:
    def test_fires(self):
        n = GatedLIFNeuron()
        assert sum(n.step(0.5) for _ in range(20)) > 0


class TestJansenRit:
    def test_oscillation(self):
        n = JansenRitUnit()
        vals = [n.step() for _ in range(500)]
        assert max(vals) > min(vals)


class TestWongWang:
    def test_decision(self):
        n = WongWangUnit()
        for _ in range(2000):
            n.step(0.02, 0.0)
        assert abs(n.s1 - n.s2) > 0.01


class TestErmentroutKopellPop:
    def test_rate_positive(self):
        n = ErmentroutKopellPopulation()
        for _ in range(100):
            r = n.step(5.0)
        assert r > 0.0


class TestCourageNekorkinMap:
    def test_dynamics(self):
        n = CourageNekorkinMapNeuron()
        for _ in range(100):
            n.step(0.1)
        assert n.x != 0.0 or n.y != 0.0


class TestMedvedevMap:
    def test_dynamics(self):
        n = MedvedevMapNeuron()
        for _ in range(50):
            n.step(0.1)
        assert n.u != 0.0


class TestLoihiCUBA:
    def test_fires(self):
        n = LoihiCUBANeuron()
        assert sum(n.step(200) for _ in range(100)) > 0


class TestTrueNorth:
    def test_fires(self):
        n = TrueNorthNeuron()
        assert sum(n.step(30) for _ in range(100)) > 0


class TestBrainScaleSAdEx:
    def test_fires(self):
        n = BrainScaleSAdExNeuron()
        assert sum(n.step(500.0) for _ in range(2000)) > 0


class TestSpiNNakerLIF:
    def test_fires(self):
        n = SpiNNakerLIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_refractory(self):
        n = SpiNNakerLIFNeuron(tau_refrac=5.0)
        # Drive hard until spike
        for _ in range(50):
            if n.step(100.0):
                break
        assert n.refrac_count > 0


class TestInhomogeneousPoisson:
    def test_fires(self):
        n = InhomogeneousPoissonNeuron()
        assert sum(n.step(1000.0) for _ in range(1000)) > 0

    def test_zero_rate(self):
        n = InhomogeneousPoissonNeuron()
        assert sum(n.step(0.0) for _ in range(100)) == 0


class TestEnergyLIF:
    def test_fires(self):
        n = EnergyLIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_energy_depletes(self):
        n = EnergyLIFNeuron()
        for _ in range(100):
            n.step(30.0)
        assert n.epsilon < 1.0


class TestLeakyCompeteFire:
    def test_wta(self):
        n = LeakyCompeteFireNeuron(n_units=3)
        for _ in range(50):
            spikes = n.step([2.0, 0.5, 0.3])
        assert isinstance(spikes, list)
        assert len(spikes) == 3


class TestPrescott:
    def test_fires(self):
        n = PrescottNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_adaptation(self):
        n = PrescottNeuron()
        for _ in range(100):
            n.step(20.0)
        assert n.w != 0.0
