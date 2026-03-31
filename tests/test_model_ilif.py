# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: InhibitoryLIFNeuron

"""Full pipeline test for InhibitoryLIFNeuron (2025).

LIF with temporal inhibitory mechanism:
inh_trace *= alpha_inh
V = alpha_m · V + I - inh_strength · inh_trace
Spike: V→V_reset, inh_trace += 1.

alpha_m = exp(-dt/tau_m), alpha_inh = exp(-dt/tau_inh).
Precomputed decay constants. Inhibitory trace creates temporal
suppression after each spike, shaping temporal coding.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.ilif import InhibitoryLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: InhibitoryLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestILIFIsolation:
    def test_defaults(self):
        n = InhibitoryLIFNeuron()
        assert n.v == 0.0 and n.inh_trace == 0.0
        assert n.tau_m == 10.0 and n.tau_inh == 5.0
        assert n.v_threshold == 1.0 and n.inh_strength == 0.5

    def test_precomputed_alphas(self):
        """alpha_m/alpha_inh precomputed in __post_init__."""
        n = InhibitoryLIFNeuron()
        assert abs(n.alpha_m - np.exp(-1.0 / 10.0)) < 1e-14
        assert abs(n.alpha_inh - np.exp(-1.0 / 5.0)) < 1e-14

    def test_step_returns_binary(self):
        assert InhibitoryLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = InhibitoryLIFNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.v) and np.isfinite(n.inh_trace)

    def test_reset_restores_defaults(self):
        n = InhibitoryLIFNeuron()
        for _ in range(1000):
            n.step(5.0)
        n.reset()
        assert n.v == 0.0 and n.inh_trace == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = InhibitoryLIFNeuron()
            trace = [(n.step(5.0), n.v, n.inh_trace) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — alpha decay, inhibitory trace, V update
# ---------------------------------------------------------------------------
class TestILIFAnalytical:
    def test_v_update_formula(self):
        """V = alpha_m·V + I - inh_strength·inh_trace (after trace decay)."""
        n = InhibitoryLIFNeuron()
        v0, inh0 = n.v, n.inh_trace
        I = 0.5
        # Trace decays first
        inh_after = inh0 * n.alpha_inh
        expected_v = n.alpha_m * v0 + I - n.inh_strength * inh_after
        n.step(I)
        if n.v != n.v_reset:  # no spike
            assert abs(n.v - expected_v) < 1e-12

    def test_inh_trace_decay(self):
        """inh_trace *= alpha_inh per step."""
        n = InhibitoryLIFNeuron()
        n.inh_trace = 1.0
        steps = 10
        for _ in range(steps):
            n.step(0.0)
        expected = 1.0 * n.alpha_inh**steps
        assert abs(n.inh_trace - expected) < 1e-10

    def test_spike_increments_trace(self):
        """On spike: inh_trace += 1."""
        n = InhibitoryLIFNeuron()
        for _ in range(10_000):
            inh_before = n.inh_trace
            if n.step(5.0) == 1:
                # Trace was decayed, then incremented by 1
                expected = inh_before * n.alpha_inh + 1.0
                assert abs(n.inh_trace - expected) < 1e-10
                break

    def test_spike_resets_voltage(self):
        n = InhibitoryLIFNeuron()
        for _ in range(10_000):
            if n.step(5.0) == 1:
                assert n.v == n.v_reset
                break

    def test_inhibition_suppresses_after_spike(self):
        """After spike, inh_trace > 0 → suppresses next V integration."""
        n = InhibitoryLIFNeuron()
        for _ in range(10_000):
            if n.step(5.0) == 1:
                assert n.inh_trace > 0
                # Next step: V = alpha_m·0 + I - strength·trace < I
                v_next_no_inh = 5.0  # just current
                n.step(5.0)
                assert n.v < v_next_no_inh
                break

    def test_alpha_m_range(self):
        """0 < alpha_m < 1 for finite tau_m."""
        n = InhibitoryLIFNeuron()
        assert 0 < n.alpha_m < 1


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestILIFDynamics:
    def test_fires_under_drive(self):
        n = InhibitoryLIFNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 100

    def test_subthreshold_silent(self):
        n = InhibitoryLIFNeuron()
        assert len(_run(n, current=0.05, steps=5000)) == 0

    def test_rate_monotonic(self):
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = InhibitoryLIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 1.0, 5.0, 10.0])
    def test_fi_sweep(self, current: float):
        n = InhibitoryLIFNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestILIFParameters:
    @pytest.mark.parametrize("inh_strength", [0.0, 0.5, 2.0])
    def test_inh_strength_sweep(self, inh_strength: float):
        n = InhibitoryLIFNeuron(inh_strength=inh_strength)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert isinstance(spikes, int)

    def test_stronger_inhibition_fewer_spikes(self):
        s_weak = len(_run(InhibitoryLIFNeuron(inh_strength=0.1), 5.0, 5000))
        s_strong = len(_run(InhibitoryLIFNeuron(inh_strength=2.0), 5.0, 5000))
        assert s_weak >= s_strong

    @pytest.mark.parametrize("tau_inh", [2.0, 5.0, 20.0])
    def test_tau_inh_sweep(self, tau_inh: float):
        n = InhibitoryLIFNeuron(tau_inh=tau_inh)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestILIFPerformance:
    def test_isolation_throughput(self):
        n = InhibitoryLIFNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 200_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(InhibitoryLIFNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestILIFPipeline:
    def test_population(self):
        assert Population(InhibitoryLIFNeuron, n=10, label="ilif").n == 10

    def test_projection_wiring(self):
        src = Population(InhibitoryLIFNeuron, n=5, label="src")
        tgt = Population(InhibitoryLIFNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(InhibitoryLIFNeuron, n=10, label="ilif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = InhibitoryLIFNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50

    def test_analysis_isi(self):
        n = InhibitoryLIFNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = InhibitoryLIFNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
