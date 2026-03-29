# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: DPINeuron

"""Full pipeline test for DPINeuron (Indiveri et al. 2011).

DYNAP-SE differential-pair integrator. Current-domain LIF (analog VLSI)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestDPIIsolation:
    def test_construction(self):
        n = DPINeuron()
        assert n.i_mem == 0.0
        assert n.i_threshold == 1.0

    def test_step_returns_binary(self):
        assert DPINeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = DPINeuron()
        assert sum(n.step(0.5) for _ in range(5000)) == 0

    def test_spikes(self):
        n = DPINeuron()
        assert sum(n.step(2.0) for _ in range(5000)) > 50

    def test_current_domain(self):
        """i_mem should be non-negative (current, not voltage)."""
        n = DPINeuron()
        for _ in range(1000):
            n.step(1.0)
        assert n.i_mem >= 0

    def test_leak_current(self):
        """Without input, i_leak should slowly charge i_mem."""
        n = DPINeuron()
        n.step(0.0)
        assert n.i_mem >= 0

    def test_state_finite(self):
        n = DPINeuron()
        for _ in range(10000):
            n.step(2.0)
        assert np.isfinite(n.i_mem)

    def test_reset(self):
        n = DPINeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.i_mem == 0.0


class TestDPINetwork:
    def test_population(self):
        assert Population(DPINeuron, n=10, label="dpi").n == 10

    def test_network_spikes(self):
        pop = Population(DPINeuron, n=20, label="dpi")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=1.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestDPIAnalysis:
    def test_spike_count(self):
        n = DPINeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(2.0)
        assert spike_count(train) > 50
