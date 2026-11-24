# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR public simulator workflow

"""Exercise Wilson-HR through populations, networks, projections, and analysis."""

from __future__ import annotations

from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron
from sc_neurocore.analysis.spike_stats.basic import firing_rate
from sc_neurocore.analysis.spike_stats.basic import isi
import numpy as np
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestWilsonHRPublicWorkflow:
    """Named workflow contract: Wilson-HR public surface inside the Python simulator."""

    def test_population(self) -> None:
        """A population of ten neurons is built."""
        assert Population(WilsonHRNeuron, n=10, label="whr").n == 10

    def test_network_spikes(self) -> None:
        """A driven network spikes."""
        pop = Population(WilsonHRNeuron, n=10, label="whr")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.1, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self) -> None:
        """A driven projection makes the network spike."""
        src = Population(WilsonHRNeuron, n=10, label="src")
        tgt = Population(WilsonHRNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.1, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.2, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self) -> None:
        """A driven run gives at least 400 spikes, several intervals and a positive rate."""
        n = WilsonHRNeuron()
        train = np.array([float(n.step(0.1)) for _ in range(50_000)])
        sc = spike_count(train)
        assert sc >= 400
        intervals = isi(train, dt=0.00005)
        assert len(intervals) >= 3
        rate = firing_rate(train, dt=0.00005)
        assert rate > 0
