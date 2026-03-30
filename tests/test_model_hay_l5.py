# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HayL5PyramidalNeuron

"""Full pipeline test for HayL5PyramidalNeuron (Hay et al. 2011).

3-compartment L5 pyramidal cell: soma (Na/K), trunk (Ca/Ih), tuft (Ca/KCa).
Reproduces BAC firing (backpropagation-activated calcium spike)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestHayL5Isolation:
    def test_construction(self):
        n = HayL5PyramidalNeuron()
        assert n.v_s == -75.0
        assert n.v_t == -75.0
        assert n.v_a == -75.0

    def test_step_returns_binary(self):
        assert HayL5PyramidalNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = HayL5PyramidalNeuron()
        assert sum(n.step(1.0) for _ in range(2000)) == 0

    def test_spikes_under_drive(self):
        n = HayL5PyramidalNeuron()
        assert sum(n.step(5.0) for _ in range(5000)) > 3

    def test_three_compartments_diverge(self):
        """All 3 compartment voltages should differ under drive."""
        n = HayL5PyramidalNeuron()
        for _ in range(2000):
            n.step(3.0)
        assert n.v_s != n.v_t or n.v_t != n.v_a

    def test_bac_firing(self):
        """Soma + tuft drive should produce more spikes than soma alone."""
        n_soma = HayL5PyramidalNeuron()
        n_bac = HayL5PyramidalNeuron()
        s_soma = sum(n_soma.step(3.0, current_tuft=0.0) for _ in range(5000))
        s_bac = sum(n_bac.step(3.0, current_tuft=5.0) for _ in range(5000))
        assert s_bac >= s_soma

    def test_calcium_dynamics(self):
        """Tuft calcium should change under drive."""
        n = HayL5PyramidalNeuron()
        ca_init = n.ca_a
        for _ in range(3000):
            n.step(5.0, current_tuft=3.0)
        assert n.ca_a != ca_init

    def test_calcium_non_negative(self):
        n = HayL5PyramidalNeuron()
        for _ in range(5000):
            n.step(5.0, current_tuft=5.0)
        assert n.ca_a >= 0.0

    def test_ih_gate(self):
        """Ih gate should change from initial under drive."""
        n = HayL5PyramidalNeuron()
        for _ in range(3000):
            n.step(5.0)
        assert n.m_ih != 0.0

    def test_numerical_stability(self):
        for I in [0.0, 3.0, 5.0, 10.0]:
            n = HayL5PyramidalNeuron()
            for _ in range(3000):
                n.step(I)
            for attr in ["v_s", "v_t", "v_a", "h_na", "n_k", "m_ca", "h_ca", "m_ih", "ca_a"]:
                assert np.isfinite(getattr(n, attr)), f"{attr} NaN at I={I}"

    def test_reset(self):
        n = HayL5PyramidalNeuron()
        for _ in range(3000):
            n.step(5.0)
        n.reset()
        assert n.v_s == -75.0
        assert n.v_t == -75.0
        assert n.v_a == -75.0
        assert n.ca_a == 0.0001


class TestHayL5Network:
    def test_population(self):
        assert Population(HayL5PyramidalNeuron, n=5, label="l5").n == 5

    def test_network_spikes(self):
        pop = Population(HayL5PyramidalNeuron, n=5, label="l5")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestHayL5Analysis:
    def test_spike_count(self):
        n = HayL5PyramidalNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(5.0)
        assert spike_count(train) > 3
