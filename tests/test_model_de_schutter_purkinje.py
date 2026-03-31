# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: DeSchutterPurkinjeNeuron

"""Full pipeline test for DeSchutterPurkinjeNeuron.

Complex Purkinje cell model. Needs very high current (I≥500) for even
1 transient spike at default params. Converges to stable fixed point.
Performance: ~4.8K steps/s (complex multi-current model)."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.de_schutter_purkinje import DeSchutterPurkinjeNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _run(neuron: DeSchutterPurkinjeNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestDeSchutterIsolation:
    def test_step_returns_binary(self):
        assert DeSchutterPurkinjeNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = DeSchutterPurkinjeNeuron()
        for _ in range(20000):
            n.step(10.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = DeSchutterPurkinjeNeuron()
        for _ in range(100):
            n.step(10.0)
        n.reset()
        assert np.isfinite(n.v)


class TestDeSchutterDynamics:
    def test_converges_to_fixed_point(self):
        """V converges to stable FP at I=0."""
        n = DeSchutterPurkinjeNeuron()
        for _ in range(20000):
            n.step(0.0)
        v1 = n.v
        for _ in range(10000):
            n.step(0.0)
        assert abs(n.v - v1) < 0.1

    def test_v_shifts_with_current(self):
        n0 = DeSchutterPurkinjeNeuron()
        n100 = DeSchutterPurkinjeNeuron()
        for _ in range(20000):
            n0.step(0.0)
            n100.step(100.0)
        assert n100.v > n0.v

    def test_high_current_transient_spike(self):
        """I=500+ can produce 1 transient spike."""
        n = DeSchutterPurkinjeNeuron()
        spikes = _run(n, current=500.0, steps=20000)
        assert len(spikes) >= 1

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = DeSchutterPurkinjeNeuron()
            trace = [(n.step(10.0), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestDeSchutterPerformance:
    def test_isolation_throughput(self):
        n = DeSchutterPurkinjeNeuron()
        N = 2000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 1000


class TestDeSchutterPipeline:
    def test_population(self):
        assert Population(DeSchutterPurkinjeNeuron, n=3, label="dsp").n == 3

    def test_network_runs(self):
        pop = Population(DeSchutterPurkinjeNeuron, n=3, label="dsp")
        drive = PoissonInput(n=3, rate_hz=100.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
