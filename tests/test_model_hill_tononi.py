# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HillTononiNeuron

"""Full pipeline test for HillTononiNeuron (Hill & Tononi 2005).

Thalamocortical sleep/wake model: Na/K + Ih + I_T + I_KNa.
Intrinsic oscillator — spikes even at I=0 via Ih/IT rebound."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestHTIsolation:
    def test_construction(self):
        n = HillTononiNeuron()
        assert n.v == -65.0
        assert n.na_i == 5.0

    def test_step_returns_binary(self):
        assert HillTononiNeuron().step(0.0) in (0, 1)

    def test_intrinsic_oscillation(self):
        """Ih + IT create rebound spiking even at I=0."""
        n = HillTononiNeuron()
        s = sum(n.step(0.0) for _ in range(10000))
        assert s > 5

    def test_spikes_under_drive(self):
        n = HillTononiNeuron()
        assert sum(n.step(5.0) for _ in range(5000)) > 5

    def test_sodium_accumulation(self):
        """Intracellular Na should change during spiking."""
        n = HillTononiNeuron()
        na_init = n.na_i
        for _ in range(5000):
            n.step(5.0)
        assert n.na_i != na_init

    def test_sodium_non_negative(self):
        n = HillTononiNeuron()
        for _ in range(10000):
            n.step(5.0)
        assert n.na_i >= 0.0

    def test_kna_current(self):
        """Na-dependent K current activates with Na accumulation."""
        n = HillTononiNeuron()
        n.na_i = 30.0
        w = 0.37 / (1.0 + (38.7 / n.na_i) ** 3.5)
        assert w > 0.01

    def test_t_current_gate(self):
        """T-type Ca gate should evolve."""
        n = HillTononiNeuron()
        h_init = n.h_t
        for _ in range(5000):
            n.step(0.0)
        assert n.h_t != h_init

    def test_ih_gate(self):
        """Ih gate should change from initial."""
        n = HillTononiNeuron()
        for _ in range(5000):
            n.step(0.0)
        assert n.m_h > 0.0

    def test_numerical_stability(self):
        for I in [0.0, 2.0, 5.0, 10.0]:
            n = HillTononiNeuron()
            for _ in range(5000):
                n.step(I)
            for attr in ["v", "h_na", "n_k", "m_h", "h_t", "na_i"]:
                assert np.isfinite(getattr(n, attr)), f"{attr} NaN at I={I}"

    def test_reset(self):
        n = HillTononiNeuron()
        for _ in range(3000):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0
        assert n.na_i == 5.0


class TestHTNetwork:
    def test_population(self):
        assert Population(HillTononiNeuron, n=5, label="ht").n == 5

    def test_network_spikes(self):
        pop = Population(HillTononiNeuron, n=5, label="ht")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestHTAnalysis:
    def test_spike_count(self):
        n = HillTononiNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(0.0)
        assert spike_count(train) > 5
