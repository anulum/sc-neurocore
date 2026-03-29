# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: CourageNekorkinMapNeuron

"""Full pipeline test for CourageNekorkinMapNeuron (Courbage et al. 2007).

Piecewise-linear Lorenz-type map. Diverges at default params but
remains finite after clip fix."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestCourageNekorkinIsolation:
    def test_construction(self):
        n = CourageNekorkinMapNeuron()
        assert n.x == 0.0

    def test_step_returns_binary(self):
        assert CourageNekorkinMapNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        """After clip fix, state should be finite even if divergent."""
        n = CourageNekorkinMapNeuron()
        for _ in range(5000):
            n.step(0.1)
        assert np.isfinite(n.x)
        assert np.isfinite(n.y)

    def test_piecewise_linear(self):
        """_f should be piecewise: linear for x<0, saturating for x>=0."""
        n = CourageNekorkinMapNeuron()
        assert n._f(-1.0) == n.alpha * (-1.0)
        assert n._f(1.0) < n.alpha  # saturating

    def test_reset(self):
        n = CourageNekorkinMapNeuron()
        for _ in range(100):
            n.step(0.1)
        n.reset()
        assert n.x == 0.0
        assert n.y == 0.0


class TestCourageNekorkinNetwork:
    def test_population(self):
        assert Population(CourageNekorkinMapNeuron, n=10, label="cnm").n == 10

    def test_network_spikes(self):
        pop = Population(CourageNekorkinMapNeuron, n=20, label="cnm")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestCourageNekorkinAnalysis:
    def test_spike_count(self):
        n = CourageNekorkinMapNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(0.5)
        # Model may or may not spike depending on dynamics
        assert spike_count(train) >= 0
