# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeMonitor from former test_network_monitors_stimulus.py

"""Focused suite: TestSpikeMonitor from former test_network_monitors_stimulus.py."""

from __future__ import annotations

from tests.network_monitors_stimulus_support import *  # noqa: F403


class TestSpikeMonitor:
    """Spike recording, trains and labelling of SpikeMonitor."""

    def test_empty_after_init(self) -> None:
        """A new spike monitor has a count of zero."""
        pop = Population(StochasticLIFNeuron, n=10, label="exc")
        mon = SpikeMonitor(pop)
        assert mon.count == 0

    def test_record_single_spike(self) -> None:
        """Recording one spike raises the count to one."""
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        mon = SpikeMonitor(pop)
        spikes = np.array([0, 1, 0, 0, 0], dtype=np.int8)
        mon.record(spikes, t_step=10)
        assert mon.count == 1

    def test_record_multiple_spikes(self) -> None:
        """Spikes recorded over two steps are all counted."""
        pop = Population(StochasticLIFNeuron, n=4, label="test")
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 1, 0, 1]), t_step=0)
        mon.record(np.array([0, 0, 1, 0]), t_step=1)
        assert mon.count == 4

    def test_spike_trains_dict(self) -> None:
        """Spike trains are a dictionary keyed by neuron with one entry per spike."""
        pop = Population(StochasticLIFNeuron, n=3, label="test")
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0, 0]), t_step=5)
        mon.record(np.array([1, 0, 1]), t_step=10)
        trains = mon.spike_trains
        assert isinstance(trains, dict)
        assert 0 in trains
        assert len(trains[0]) == 2  # neuron 0 spiked at step 5 and 10

    def test_label(self) -> None:
        """An explicit monitor label is kept."""
        pop = Population(StochasticLIFNeuron, n=5, label="my_pop")
        mon = SpikeMonitor(pop, label="custom_label")
        assert mon.label == "custom_label"

    def test_no_spikes_empty_trains(self) -> None:
        """Recording only silent steps leaves the count at zero."""
        pop = Population(StochasticLIFNeuron, n=3, label="quiet")
        mon = SpikeMonitor(pop)
        for t in range(50):
            mon.record(np.zeros(3, dtype=np.int8), t_step=t)
        assert mon.count == 0
