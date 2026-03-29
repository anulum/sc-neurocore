# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AkidaNeuron

"""Full pipeline test for AkidaNeuron (BrainChip Akida 2021).

Event-domain rank-order IF neuron. Fires at most ONCE per presentation
(first-to-spike competition). Integer membrane potential."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.akida_neuron import AkidaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestAkidaIsolation:
    def test_construction(self):
        n = AkidaNeuron()
        assert n.v == 0
        assert n.threshold == 100

    def test_step_returns_binary(self):
        n = AkidaNeuron()
        assert n.step(0) in (0, 1)

    def test_spikes_under_drive(self):
        n = AkidaNeuron()
        spikes = 0
        for _ in range(20):
            spikes += n.step(30)
        assert spikes == 1, "Akida should fire exactly once"

    def test_fires_only_once(self):
        """Akida fires at most once per presentation (_spiked flag)."""
        n = AkidaNeuron()
        total = sum(n.step(50) for _ in range(100))
        assert total == 1

    def test_rank_order_modulation(self):
        """Later events contribute less (modulation^rank decay)."""
        n = AkidaNeuron()
        n.step(30)
        v_after_first = n.v
        n_fresh = AkidaNeuron()
        n_fresh._rank = 5
        n_fresh.step(30)
        v_after_rank5 = n_fresh.v
        assert v_after_first > v_after_rank5, "rank decay not working"

    def test_zero_weight_no_accumulation(self):
        n = AkidaNeuron()
        n.step(0)
        assert n.v == 0
        assert n._rank == 0

    def test_reset(self):
        n = AkidaNeuron()
        for _ in range(10):
            n.step(30)
        n.reset()
        assert n.v == 0
        assert n._rank == 0
        assert not n._spiked

    def test_state_integer(self):
        n = AkidaNeuron()
        for _ in range(5):
            n.step(25)
        assert isinstance(n.v, int)


class TestAkidaNetwork:
    def test_population(self):
        pop = Population(AkidaNeuron, n=10, label="akida")
        assert pop.n == 10
        assert pop.model_name == "AkidaNeuron"

    def test_network_spikes(self):
        pop = Population(AkidaNeuron, n=10, label="akida")
        drive = PoissonInput(n=10, rate_hz=1000.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.1, dt=0.001, backend="python")
        # Each neuron fires at most once → max 10 spikes
        assert 0 < mon.count <= 10

    def test_first_to_spike_property(self):
        """Each neuron should fire at most once (first-to-spike)."""
        pop = Population(AkidaNeuron, n=20, label="akida")
        drive = PoissonInput(n=20, rate_hz=1000.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.2, dt=0.001, backend="python")
        trains = mon.spike_trains
        for nid, times in trains.items():
            assert len(times) <= 1, f"neuron {nid} fired {len(times)} times"


class TestAkidaAnalysis:
    def test_spike_count(self):
        n = AkidaNeuron()
        train = np.zeros(100, dtype=np.int8)
        for t in range(100):
            train[t] = n.step(40)  # need enough weight to cross threshold=100
        assert spike_count(train) == 1
