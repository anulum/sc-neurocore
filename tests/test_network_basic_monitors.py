# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMonitors from former test_network_basic.py

"""Focused suite: TestMonitors from former test_network_basic.py."""

from __future__ import annotations

from tests.network_basic_support import *  # noqa: F403

class TestMonitors:
    def test_spike_monitor_count(self):
        pop = Population("LapicqueNeuron", 3)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0, 1], dtype=np.int8), 0)
        mon.record(np.array([0, 1, 0], dtype=np.int8), 1)
        assert mon.count == 3

    def test_spike_monitor_trains(self):
        pop = Population("LapicqueNeuron", 2)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0], dtype=np.int8), 0)
        mon.record(np.array([1, 1], dtype=np.int8), 5)
        trains = mon.spike_trains
        assert len(trains[0]) == 2
        assert len(trains[1]) == 1

    def test_spike_monitor_raster(self):
        pop = Population("LapicqueNeuron", 2)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0], dtype=np.int8), 0)
        ts, ids = mon.raster_data()
        assert len(ts) == 1
        assert ids[0] == 0

    def test_spike_monitor_firing_rates(self):
        pop = Population("LapicqueNeuron", 2)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0], dtype=np.int8), 0)
        mon.record(np.array([1, 0], dtype=np.int8), 1)
        rates = mon.firing_rates(n_steps=100, dt=0.001)
        assert rates[0] > 0
        assert rates[1] == 0.0

    def test_spike_monitor_isi(self):
        pop = Population("LapicqueNeuron", 1)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1], dtype=np.int8), 10)
        mon.record(np.array([1], dtype=np.int8), 20)
        mon.record(np.array([1], dtype=np.int8), 35)
        intervals = mon.isi(0)
        np.testing.assert_array_equal(intervals, [10, 15])

    def test_state_monitor_traces(self):
        pop = Population("LapicqueNeuron", 2)
        mon = StateMonitor(pop, variables=["v"])
        mon.snapshot(0)
        mon.snapshot(1)
        assert mon.traces["v"].shape == (2, 2)
        assert len(mon.t) == 2

    def test_rate_monitor(self):
        pop = Population("LapicqueNeuron", 4)
        mon = RateMonitor(pop, bin_ms=10)
        for t in range(100):
            spikes = np.array([1, 0, 0, 0], dtype=np.int8)
            mon.record(spikes, t, dt=0.001)
        assert len(mon.rate) > 0
        assert mon.rate[0] > 0
