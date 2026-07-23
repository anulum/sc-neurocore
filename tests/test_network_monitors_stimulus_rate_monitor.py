# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRateMonitor from former test_network_monitors_stimulus.py

"""Focused suite: TestRateMonitor from former test_network_monitors_stimulus.py."""

from __future__ import annotations

from tests.network_monitors_stimulus_support import *  # noqa: F403

class TestRateMonitor:
    def test_empty_after_init(self):
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        mon = RateMonitor(pop, bin_ms=10)
        assert len(mon._spike_counts) == 0

    def test_bin_accumulation(self):
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        mon = RateMonitor(pop, bin_ms=10)
        # 10 ms bin at dt=1ms = 10 steps
        for t in range(20):
            spikes = np.array([1 if t % 5 == 0 else 0] * 10)
            mon.record(spikes, t_step=t, dt=0.001)
        # After 20 steps (2 bins of 10 ms)
        assert len(mon._spike_counts) >= 1
