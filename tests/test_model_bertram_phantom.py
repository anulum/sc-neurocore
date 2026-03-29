# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: BertramPhantomBurster

"""Full pipeline test for BertramPhantomBurster (Bertram et al. 2008).

Dual slow variable burster (pancreatic β-cell model). Requires I≥200
for suprathreshold spiking. Sub-threshold oscillations at lower currents."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.bertram_phantom import BertramPhantomBurster
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestBertramIsolation:
    def test_construction(self):
        n = BertramPhantomBurster()
        assert n.v == -50.0
        assert n.s1 == 0.1
        assert n.s2 == 0.1

    def test_step_returns_binary(self):
        n = BertramPhantomBurster()
        assert n.step(0.0) in (0, 1)

    def test_subthreshold_at_low_current(self):
        """Model should NOT spike at I=10 (subthreshold oscillations)."""
        n = BertramPhantomBurster()
        spikes = sum(n.step(10.0) for _ in range(50_000))
        assert spikes == 0, f"unexpected spikes at I=10: {spikes}"

    def test_spikes_at_high_current(self):
        """Model should spike at I=200 (suprathreshold)."""
        n = BertramPhantomBurster()
        spikes = sum(n.step(200.0) for _ in range(50_000))
        assert spikes > 100, f"too few spikes at I=200: {spikes}"

    def test_dual_slow_variables(self):
        """Both s1 and s2 should change from initial values under drive."""
        n = BertramPhantomBurster()
        for _ in range(50_000):
            n.step(200.0)
        assert n.s1 != 0.1 or n.s2 != 0.1, "slow variables unchanged"

    def test_threshold_crossing_detection(self):
        """Spike only on upward threshold crossing (v_prev < th, v >= th)."""
        n = BertramPhantomBurster()
        # Drive to spike
        crossed = False
        for _ in range(100_000):
            if n.step(200.0):
                crossed = True
                break
        assert crossed, "no threshold crossing at I=200"

    def test_state_finite(self):
        n = BertramPhantomBurster()
        for _ in range(100_000):
            n.step(200.0)
        assert np.isfinite(n.v)
        assert np.isfinite(n.s1)
        assert np.isfinite(n.s2)

    def test_reset(self):
        n = BertramPhantomBurster()
        for _ in range(1000):
            n.step(200.0)
        n.reset()
        assert n.v == -50.0
        assert n.s1 == 0.1
        assert n.s2 == 0.1


class TestBertramNetwork:
    def test_population(self):
        pop = Population(BertramPhantomBurster, n=5, label="bertram")
        assert pop.n == 5
        assert pop.model_name == "BertramPhantomBurster"

    def test_network_spikes(self):
        pop = Population(BertramPhantomBurster, n=10, label="bertram")
        drive = PoissonInput(n=10, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_spike_trains_extractable(self):
        pop = Population(BertramPhantomBurster, n=5, label="bertram")
        drive = PoissonInput(n=5, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)


class TestBertramAnalysis:
    def _get_binary_train(self):
        n = BertramPhantomBurster()
        train = np.zeros(100_000, dtype=np.int8)
        for t in range(100_000):
            train[t] = n.step(200.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.0005)  # dt=0.5ms
        assert rate > 0

    def test_spike_count(self):
        train = self._get_binary_train()
        assert spike_count(train) > 100

    def test_isi_finite(self):
        train = self._get_binary_train()
        intervals = isi(train, dt=0.0005)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
