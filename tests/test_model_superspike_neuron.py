# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SuperSpikeNeuron

"""Full pipeline test for SuperSpikeNeuron (Zenke & Ganguli 2018).

LIF with surrogate gradient σ'(V) = 1/(β|V-θ|+1)² and Van Rossum
eligibility trace. Designed for gradient-based SNN training.
Performance: ~328K isolation steps/s."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.superspike_neuron import SuperSpikeNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: SuperSpikeNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestSuperSpikeIsolation:
    def test_construction_defaults(self):
        n = SuperSpikeNeuron()
        assert n.v == 0.0
        assert n.trace == 0.0
        assert n.tau_m == 10.0
        assert n.tau_e == 10.0
        assert n.v_threshold == 1.0
        assert n.beta_sg == 10.0

    def test_alpha_precomputed(self):
        n = SuperSpikeNeuron()
        assert abs(n.alpha_m - np.exp(-1.0 / 10.0)) < 1e-12
        assert abs(n.alpha_e - np.exp(-1.0 / 10.0)) < 1e-12

    def test_step_returns_binary(self):
        assert SuperSpikeNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = SuperSpikeNeuron()
        for _ in range(50000):
            n.step(0.2)
        assert np.isfinite(n.v) and np.isfinite(n.trace)

    def test_reset(self):
        n = SuperSpikeNeuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0 and n.trace == 0.0


class TestSuperSpikeSurrogateGradient:
    """Core: σ'(V) = 1/(β|V-θ|+1)². Peaks at V=θ, decays with distance."""

    def test_sg_peak_at_threshold(self):
        """σ'(θ) = 1/(0+1)² = 1.0 — maximum at threshold."""
        n = SuperSpikeNeuron()
        n.v = n.v_threshold
        assert abs(n.surrogate_grad() - 1.0) < 1e-10

    def test_sg_symmetric_around_threshold(self):
        """σ'(θ+δ) = σ'(θ-δ) — symmetric in |V-θ|."""
        n = SuperSpikeNeuron()
        for delta in [0.1, 0.5, 1.0, 5.0]:
            n.v = n.v_threshold + delta
            sg_above = n.surrogate_grad()
            n.v = n.v_threshold - delta
            sg_below = n.surrogate_grad()
            assert abs(sg_above - sg_below) < 1e-10, f"delta={delta}"

    def test_sg_decays_with_distance(self):
        """σ' decreases as |V-θ| increases."""
        n = SuperSpikeNeuron()
        sgs = []
        for v in [1.0, 0.9, 0.5, 0.0, -1.0]:
            n.v = v
            sgs.append(n.surrogate_grad())
        # Should be monotonically decreasing
        assert all(sgs[i] >= sgs[i + 1] for i in range(len(sgs) - 1))

    def test_sg_formula_exact(self):
        """Verify σ' = 1/(β|V-θ|+1)² at specific V."""
        n = SuperSpikeNeuron()
        n.v = 0.5  # |V-θ| = 0.5
        expected = 1.0 / (n.beta_sg * 0.5 + 1.0) ** 2
        assert abs(n.surrogate_grad() - expected) < 1e-10

    def test_beta_controls_sharpness(self):
        """Higher beta → sharper peak (faster decay away from θ)."""
        n_sharp = SuperSpikeNeuron(beta_sg=50.0)
        n_soft = SuperSpikeNeuron(beta_sg=1.0)
        # At V = θ - 0.5:
        n_sharp.v = 0.5
        n_soft.v = 0.5
        assert n_soft.surrogate_grad() > n_sharp.surrogate_grad()


class TestSuperSpikeEligibilityTrace:
    """trace = α_e · trace + σ'(V). Leaky integrator of surrogate gradient."""

    def test_trace_accumulates_sg(self):
        """Trace grows when σ'(V) > 0 (always, but peaks near threshold)."""
        n = SuperSpikeNeuron()
        t0 = n.trace
        n.step(0.5)
        assert n.trace > t0

    def test_trace_decays_without_sg(self):
        """With V far from threshold: σ' ≈ 0, trace decays."""
        n = SuperSpikeNeuron()
        n.trace = 5.0
        n.v = -10.0  # far from threshold → σ' ≈ 0
        n.step(0.0)  # v stays negative, σ' tiny
        assert n.trace < 5.0

    def test_trace_peaks_near_threshold(self):
        """When V hovers near threshold, trace accumulates fastest."""
        # Near threshold: higher SG → faster trace growth
        n_near = SuperSpikeNeuron()
        n_far = SuperSpikeNeuron()
        for _ in range(100):
            n_near.step(0.09)  # v ≈ 0.9, near threshold
            n_far.step(0.001)  # v ≈ 0.01, far from threshold
        assert n_near.trace > n_far.trace


class TestSuperSpikeLIFDynamics:
    def test_voltage_leaky_integration(self):
        """v = alpha_m · v + I. Standard LIF with precomputed alpha."""
        n = SuperSpikeNeuron(v_threshold=100.0)
        n.step(0.5)
        assert abs(n.v - 0.5) < 1e-10  # v = alpha*0 + 0.5 = 0.5

    def test_spike_at_threshold(self):
        n = SuperSpikeNeuron()
        n.v = 0.9
        s = n.step(0.2)  # v = alpha*0.9 + 0.2 ≈ 1.014 ≥ 1.0
        assert s == 1

    def test_reset_on_spike(self):
        n = SuperSpikeNeuron()
        n.v = 0.9
        n.step(0.2)  # spike
        assert n.v == n.v_reset


class TestSuperSpikeFI:
    def test_zero_silent(self):
        n = SuperSpikeNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.1, 0.2, 0.5, 1.0]:
            n = SuperSpikeNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


class TestSuperSpikePerformance:
    def test_isolation_throughput(self):
        n = SuperSpikeNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.2)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(SuperSpikeNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestSuperSpikePipeline:
    def test_population(self):
        assert Population(SuperSpikeNeuron, n=10, label="ss").n == 10

    def test_network_with_drive(self):
        pop = Population(SuperSpikeNeuron, n=10, label="ss")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(SuperSpikeNeuron, n=10, label="src")
        tgt = Population(SuperSpikeNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_pipeline(self):
        n = SuperSpikeNeuron()
        train = np.array([float(n.step(0.2)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SuperSpikeNeuron()
            trace = [(n.step(0.2), n.v, n.trace) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
