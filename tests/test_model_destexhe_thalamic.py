# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: DestexheThalamicNeuron

"""Full pipeline test for DestexheThalamicNeuron (Destexhe 1993).

Thalamocortical relay with T-type Ca²⁺ current. Produces rebound spikes."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.destexhe_thalamic import DestexheThalamicNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestDestexheIsolation:
    def test_construction(self):
        n = DestexheThalamicNeuron()
        assert n.v == -65.0
        assert n.h_t == 1.0

    def test_step_returns_binary(self):
        assert DestexheThalamicNeuron().step(0.0) in (0, 1)

    def test_rebound_spike(self):
        """T-current should produce at least one rebound spike."""
        n = DestexheThalamicNeuron()
        spikes = sum(n.step(5.0) for _ in range(5000))
        assert spikes >= 1, "no rebound spike"

    def test_t_current_gating(self):
        """h_t should change from initial under drive."""
        n = DestexheThalamicNeuron()
        h_init = n.h_t
        for _ in range(5000):
            n.step(5.0)
        assert n.h_t != h_init

    def test_numerical_stability(self):
        for I in [0, 5, 10]:
            n = DestexheThalamicNeuron()
            for _ in range(5000):
                n.step(float(I))
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.h_na), f"h_na NaN at I={I}"
            assert np.isfinite(n.n_k), f"n_k NaN at I={I}"
            assert np.isfinite(n.h_t), f"h_t NaN at I={I}"

    def test_reset(self):
        n = DestexheThalamicNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0
        assert n.h_t == 1.0


class TestDestexheNetwork:
    def test_population(self):
        assert Population(DestexheThalamicNeuron, n=5, label="dest").n == 5

    def test_network_spikes(self):
        pop = Population(DestexheThalamicNeuron, n=10, label="dest")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestDestexheAnalysis:
    def test_spike_count(self):
        n = DestexheThalamicNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(5.0)
        assert spike_count(train) >= 1
