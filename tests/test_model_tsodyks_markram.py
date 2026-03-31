# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: TsodyksMarkramNeuron

"""Full pipeline test for TsodyksMarkramNeuron (Tsodyks & Markram 1997).

LIF + short-term synaptic plasticity: x (depression) and u (facilitation).
step(current, presynaptic_spike). STP dynamics only on presynaptic spike.
Performance: ~615K isolation steps/s."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.tsodyks_markram import TsodyksMarkramNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: TsodyksMarkramNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestTMIsolation:
    def test_defaults(self):
        n = TsodyksMarkramNeuron()
        assert n.v == -65.0 and n.x == 1.0 and n.u == 0.2
        assert n.tau_d == 200.0 and n.tau_f == 600.0

    def test_step_returns_binary(self):
        assert TsodyksMarkramNeuron().step(0.0) in (0, 1)

    def test_dual_input_signature(self):
        """step(current, presynaptic_spike) — two inputs."""
        n = TsodyksMarkramNeuron()
        s = n.step(10.0, presynaptic_spike=True)
        assert s in (0, 1)

    def test_state_finite(self):
        n = TsodyksMarkramNeuron()
        for _ in range(50000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.x) and np.isfinite(n.u)

    def test_reset(self):
        n = TsodyksMarkramNeuron()
        for _ in range(100):
            n.step(20.0, presynaptic_spike=True)
        n.reset()
        assert n.v == n.v_rest and n.x == 1.0 and n.u == n.u_se


class TestTMSTPDynamics:
    """Short-term plasticity: x (depression) and u (facilitation)."""

    def test_x_depletes_on_presyn_spike(self):
        """x decreases on presynaptic spike (depression)."""
        n = TsodyksMarkramNeuron()
        x0 = n.x
        n.step(0.0, presynaptic_spike=True)
        assert n.x < x0, f"x={n.x}, expected depletion"

    def test_x_recovers_between_spikes(self):
        """x recovers toward 1.0 with tau_d between spikes."""
        n = TsodyksMarkramNeuron()
        n.step(0.0, presynaptic_spike=True)  # deplete
        x_after_spike = n.x
        for _ in range(2000):
            n.step(0.0)  # no presyn spike, x recovers
        assert n.x > x_after_spike

    def test_u_facilitates_on_presyn_spike(self):
        """u increases on presynaptic spike (facilitation)."""
        n = TsodyksMarkramNeuron()
        u0 = n.u
        n.step(0.0, presynaptic_spike=True)
        assert n.u > u0

    def test_u_decays_between_spikes(self):
        """u decays toward u_se with tau_f between spikes."""
        n = TsodyksMarkramNeuron()
        n.step(0.0, presynaptic_spike=True)  # facilitate
        u_after = n.u
        for _ in range(5000):
            n.step(0.0)
        assert abs(n.u - n.u_se) < abs(u_after - n.u_se)

    def test_depression_reduces_efficacy(self):
        """Repeated presyn spikes deplete x → weaker synaptic current."""
        n = TsodyksMarkramNeuron()
        # First spike: high x
        n.step(0.0, presynaptic_spike=True)
        x1 = n.x
        # Second spike: lower x
        n.step(0.0, presynaptic_spike=True)
        x2 = n.x
        assert x2 < x1, "x should deplete further on second spike"

    def test_x_bounded_0_1(self):
        """x stays in [0, 1]."""
        n = TsodyksMarkramNeuron()
        for _ in range(1000):
            n.step(0.0, presynaptic_spike=(np.random.random() < 0.5))
        assert 0.0 <= n.x <= 1.0

    def test_u_bounded_0_1(self):
        """u stays in [0, 1]."""
        n = TsodyksMarkramNeuron()
        for _ in range(1000):
            n.step(0.0, presynaptic_spike=(np.random.random() < 0.5))
        assert 0.0 <= n.u <= 1.0


class TestTMSynapticCurrent:
    def test_isyn_on_presyn_spike(self):
        """I_syn = A · u · x when presynaptic spike occurs."""
        n = TsodyksMarkramNeuron()
        # At first spike: u = u_se + U*(1-u) = 0.2 + 0.2*0.8 = 0.36
        # x starts at 1.0
        # i_syn = 50 * 0.36 * 1.0 = 18.0
        # This drives V
        v_before = n.v
        n.step(0.0, presynaptic_spike=True)
        # V should have increased from i_syn
        assert n.v > v_before

    def test_no_isyn_without_presyn(self):
        """Without presynaptic spike, I_syn = 0."""
        n = TsodyksMarkramNeuron()
        n.step(0.0, presynaptic_spike=False)
        # V should have moved only from leak (toward rest)
        assert abs(n.v - n.v_rest) < 0.01


class TestTMFI:
    def test_subthreshold_silent(self):
        n = TsodyksMarkramNeuron()
        assert len(_run(n, current=10.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = TsodyksMarkramNeuron()
        assert len(_run(n, current=20.0, steps=10000)) >= 10

    def test_monotonic_fi(self):
        rates = []
        for I in [20.0, 30.0, 50.0]:
            n = TsodyksMarkramNeuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


class TestTMPerformance:
    def test_isolation_throughput(self):
        n = TsodyksMarkramNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(TsodyksMarkramNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=25.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestTMPipeline:
    def test_population(self):
        assert Population(TsodyksMarkramNeuron, n=10, label="tm").n == 10

    def test_network_spikes(self):
        pop = Population(TsodyksMarkramNeuron, n=10, label="tm")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = TsodyksMarkramNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = TsodyksMarkramNeuron()
            trace = [(n.step(20.0), n.v, n.x, n.u) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
