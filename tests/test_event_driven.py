# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.event_driven (async simulation)

from __future__ import annotations

import numpy as np

from sc_neurocore.event_driven import EventDrivenSimulator, SpikeEvent, EventStats


def _simple_chain(n=5):
    """Linear chain: 0→1→2→3→4."""
    conns = [(i, i + 1, 0.6, 1.0) for i in range(n - 1)]
    return EventDrivenSimulator(n_neurons=n, connectivity=conns, threshold=1.0, tau_mem=20.0)


class TestSpikeEvent:
    def test_ordering(self):
        e1 = SpikeEvent(time=1.0, source_id=0, target_id=1)
        e2 = SpikeEvent(time=2.0, source_id=0, target_id=1)
        assert e1 < e2


class TestEventStats:
    def test_summary(self):
        s = EventStats(total_events_processed=100, total_spikes_generated=10, max_queue_size=50)
        assert "100 events" in s.summary()


class TestEventDrivenSimulator:
    def test_no_events(self):
        sim = _simple_chain()
        spikes, stats = sim.run(100.0)
        assert len(spikes) == 0
        assert stats.total_events_processed == 0

    def test_single_spike_propagation(self):
        sim = EventDrivenSimulator(
            n_neurons=3,
            connectivity=[(0, 1, 1.5, 1.0), (1, 2, 1.5, 1.0)],
            threshold=1.0,
            tau_mem=50.0,
        )
        sim.inject_spikes([(0.0, 0)])
        spikes, stats = sim.run(10.0)
        assert stats.total_events_processed > 0
        # Neuron 1 should fire from the spike from 0
        fired_neurons = {nid for _, nid in spikes}
        assert 1 in fired_neurons

    def test_chain_propagation(self):
        sim = EventDrivenSimulator(
            n_neurons=3,
            connectivity=[(0, 1, 2.0, 0.5), (1, 2, 2.0, 0.5)],
            threshold=1.0,
        )
        sim.inject_spikes([(0.0, 0)])
        spikes, stats = sim.run(10.0)
        fired = {nid for _, nid in spikes}
        assert 1 in fired
        assert 2 in fired

    def test_refractory_period(self):
        sim = EventDrivenSimulator(
            n_neurons=2,
            connectivity=[(0, 1, 2.0, 0.0)],
            threshold=1.0,
            refractory=5.0,
        )
        # Two spikes 1ms apart — second should be blocked by refractory
        sim.inject_spikes([(0.0, 0), (1.0, 0)])
        spikes, stats = sim.run(10.0)
        neuron1_times = [t for t, n in spikes if n == 1]
        if len(neuron1_times) >= 2:
            assert neuron1_times[1] - neuron1_times[0] >= 5.0

    def test_inject_current(self):
        sim = EventDrivenSimulator(n_neurons=2, connectivity=[], threshold=1.0)
        sim.inject_current([(0.0, 0, 0.5), (1.0, 0, 0.6)])
        spikes, stats = sim.run(10.0)
        assert stats.total_events_processed == 2

    def test_speedup_estimate(self):
        sim = EventDrivenSimulator(
            n_neurons=1000,
            connectivity=[(0, i, 2.0, 0.0) for i in range(1, 10)],
            threshold=1.0,
        )
        sim.inject_spikes([(0.0, 0)])
        _, stats = sim.run(100.0)
        if stats.total_events_processed > 0:
            assert stats.speedup_vs_clockdriven > 1.0

    def test_reset(self):
        sim = _simple_chain()
        sim.inject_spikes([(0.0, 0)])
        sim.run(10.0)
        sim.reset()
        spikes, stats = sim.run(10.0)
        assert len(spikes) == 0

    def test_delayed_events(self):
        sim = EventDrivenSimulator(
            n_neurons=2,
            connectivity=[(0, 1, 2.0, 5.0)],  # 5ms delay
            threshold=1.0,
        )
        sim.inject_spikes([(0.0, 0)])
        spikes, _ = sim.run(10.0)
        neuron1_times = [t for t, n in spikes if n == 1]
        if neuron1_times:
            assert neuron1_times[0] >= 5.0  # arrived after delay

    def test_duration_cutoff(self):
        sim = EventDrivenSimulator(
            n_neurons=2,
            connectivity=[(0, 1, 2.0, 50.0)],  # arrives at t=50
            threshold=1.0,
        )
        sim.inject_spikes([(0.0, 0)])
        spikes, _ = sim.run(10.0)  # cut off at t=10
        assert len(spikes) == 0  # event at t=50 not processed

    def test_large_sparse_network(self):
        rng = np.random.RandomState(42)
        n = 500
        conns = []
        for _ in range(1000):
            s, t = rng.randint(0, n, 2)
            if s != t:
                conns.append((int(s), int(t), 0.3, float(rng.uniform(0.1, 2.0))))
        sim = EventDrivenSimulator(n_neurons=n, connectivity=conns, threshold=1.0)
        sim.inject_spikes([(0.0, i) for i in range(10)])
        spikes, stats = sim.run(50.0)
        assert stats.total_events_processed > 0
