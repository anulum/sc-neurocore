# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ComplementaryLIFNeuron

"""Full pipeline test for ComplementaryLIFNeuron (ICML 2024).

Dual positive/negative membrane paths. Spike when |v_pos - v_neg| ≥ θ.
Ternary output {-1, 0, +1}. Both paths decay with alpha = exp(-dt/tau).
Reset zeros both paths on spike. Performance benchmarked."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.clif import ComplementaryLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: ComplementaryLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) != 0]


class TestCLIFIsolation:
    def test_construction_defaults(self):
        n = ComplementaryLIFNeuron()
        assert n.v_pos == 0.0
        assert n.v_neg == 0.0
        assert n.tau == 10.0
        assert n.v_threshold == 1.0
        assert n.dt == 1.0

    def test_alpha_property(self):
        """alpha = exp(-dt/tau). Verified analytically."""
        n = ComplementaryLIFNeuron()
        expected = np.exp(-n.dt / n.tau)
        assert abs(n.alpha - expected) < 1e-12

    def test_ternary_output(self):
        """Returns -1, 0, or +1."""
        n = ComplementaryLIFNeuron()
        assert n.step(0.0) == 0
        n2 = ComplementaryLIFNeuron()
        assert n2.step(2.0) == 1  # v_pos = 2 ≥ θ
        n3 = ComplementaryLIFNeuron()
        assert n3.step(-2.0) == -1  # v_neg = 2, diff = -2 ≤ -θ

    def test_state_finite(self):
        n = ComplementaryLIFNeuron()
        for _ in range(100000):
            n.step(1.0)
        assert np.isfinite(n.v_pos) and np.isfinite(n.v_neg)

    def test_reset(self):
        n = ComplementaryLIFNeuron()
        for _ in range(50):
            n.step(0.5)
        n.reset()
        assert n.v_pos == 0.0 and n.v_neg == 0.0


class TestCLIFDualPathMechanism:
    """Core: separate v_pos and v_neg accumulation with leaky decay."""

    def test_positive_input_charges_v_pos_only(self):
        """I > 0 → inp_pos = I, inp_neg = 0. Only v_pos accumulates."""
        n = ComplementaryLIFNeuron(v_threshold=100.0)  # prevent spikes
        n.step(0.5)
        assert n.v_pos == 0.5
        assert n.v_neg == 0.0

    def test_negative_input_charges_v_neg_only(self):
        """I < 0 → inp_pos = 0, inp_neg = |I|. Only v_neg accumulates."""
        n = ComplementaryLIFNeuron(v_threshold=100.0)
        n.step(-0.5)
        assert n.v_pos == 0.0
        assert n.v_neg == 0.5

    def test_both_paths_decay(self):
        """Both v_pos and v_neg decay with alpha each step."""
        n = ComplementaryLIFNeuron(v_threshold=100.0)
        n.v_pos = 1.0
        n.v_neg = 1.0
        n.step(0.0)  # zero input, just decay
        assert abs(n.v_pos - n.alpha) < 1e-10
        assert abs(n.v_neg - n.alpha) < 1e-10

    def test_spike_on_difference(self):
        """Spike when v_pos - v_neg ≥ θ (positive) or ≤ -θ (negative)."""
        n = ComplementaryLIFNeuron(v_threshold=1.0)
        # Positive: v_pos = 1.5 ≥ 1.0 → spike +1
        assert n.step(1.5) == 1
        # Negative
        n2 = ComplementaryLIFNeuron(v_threshold=1.0)
        assert n2.step(-1.5) == -1

    def test_reset_zeros_both_on_positive_spike(self):
        n = ComplementaryLIFNeuron()
        n.step(2.0)  # spike +1
        assert n.v_pos == 0.0 and n.v_neg == 0.0

    def test_reset_zeros_both_on_negative_spike(self):
        n = ComplementaryLIFNeuron()
        n.step(-2.0)  # spike -1
        assert n.v_pos == 0.0 and n.v_neg == 0.0

    def test_mixed_input_cancellation(self):
        """Alternating +/- input: both paths charge equally → diff ≈ 0 → no spike."""
        n = ComplementaryLIFNeuron(v_threshold=1.0)
        spikes = 0
        for t in range(1000):
            s = n.step(0.5 if t % 2 == 0 else -0.5)
            if s != 0:
                spikes += 1
        # With balanced input, diff stays small
        assert spikes < 10, f"{spikes} spikes with balanced input"


class TestCLIFSpikeRate:
    def test_rate_proportional_to_input(self):
        """For constant I < θ: rate depends on how fast v_pos accumulates to θ."""
        n1 = ComplementaryLIFNeuron()
        n2 = ComplementaryLIFNeuron()
        s1 = len(_run(n1, current=0.3, steps=5000))
        s2 = len(_run(n2, current=0.6, steps=5000))
        assert s2 > s1

    def test_suprathreshold_fires_every_step(self):
        """I ≥ θ → fires every step (v_pos immediately ≥ θ after reset)."""
        n = ComplementaryLIFNeuron()
        spikes = [n.step(1.5) for _ in range(100)]
        assert spikes.count(1) == 100

    def test_zero_input_silent(self):
        n = ComplementaryLIFNeuron()
        assert all(n.step(0.0) == 0 for _ in range(1000))

    def test_negative_input_produces_negative_spikes(self):
        n = ComplementaryLIFNeuron()
        outputs = [n.step(-1.5) for _ in range(100)]
        assert outputs.count(-1) == 100
        assert outputs.count(1) == 0


class TestCLIFAnalyticalProperties:
    def test_v_pos_steady_state(self):
        """For constant I > 0 (subthreshold): v_pos → I / (1 - alpha)."""
        n = ComplementaryLIFNeuron(v_threshold=100.0)  # no spikes
        I = 0.3
        for _ in range(1000):
            n.step(I)
        v_ss = I / (1.0 - n.alpha)
        assert abs(n.v_pos - v_ss) < 0.01, f"v_pos={n.v_pos:.4f}, V_ss={v_ss:.4f}"

    def test_alpha_tau_relationship(self):
        """Different tau → different alpha → different decay rate."""
        n_fast = ComplementaryLIFNeuron(tau=5.0)
        n_slow = ComplementaryLIFNeuron(tau=50.0)
        assert n_fast.alpha < n_slow.alpha  # faster decay → smaller alpha


class TestCLIFParameters:
    @pytest.mark.parametrize("tau", [2.0, 10.0, 50.0])
    def test_tau_variations(self, tau: float):
        n = ComplementaryLIFNeuron(tau=tau)
        for _ in range(5000):
            n.step(0.5)
        assert np.isfinite(n.v_pos)

    def test_custom_threshold(self):
        n_low = ComplementaryLIFNeuron(v_threshold=0.5)
        n_high = ComplementaryLIFNeuron(v_threshold=2.0)
        s_low = len(_run(n_low, current=0.3, steps=5000))
        s_high = len(_run(n_high, current=0.3, steps=5000))
        assert s_low > s_high

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ComplementaryLIFNeuron()
            trace = [(n.step(0.5), n.v_pos, n.v_neg) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestCLIFPerformance:
    def test_isolation_throughput(self):
        """Benchmark: isolation steps per second."""
        n = ComplementaryLIFNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        steps_per_s = N / elapsed
        # Just verify it ran and is reasonable (> 10k steps/s)
        assert steps_per_s > 10000, f"{steps_per_s:.0f} steps/s"

    def test_network_throughput(self):
        """Benchmark: network neuron-steps per second."""
        pop = Population(ComplementaryLIFNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 50 * 500
        nsteps_per_s = neuron_steps / elapsed
        assert nsteps_per_s > 1000, f"{nsteps_per_s:.0f} neuron-steps/s"


class TestCLIFPipeline:
    def test_population(self):
        assert Population(ComplementaryLIFNeuron, n=10, label="clif").n == 10

    def test_network_with_drive(self):
        pop = Population(ComplementaryLIFNeuron, n=10, label="clif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(ComplementaryLIFNeuron, n=10, label="src")
        tgt = Population(ComplementaryLIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_analysis_pipeline(self):
        n = ComplementaryLIFNeuron()
        train = np.array([float(max(0, n.step(0.5))) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 100
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 10000 * 0.001
        assert abs(rate - sc / duration) < 10.0
