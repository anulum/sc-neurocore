# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: McCullochPittsNeuron

"""Full pipeline test for McCullochPittsNeuron (McCulloch & Pitts 1943).

The first mathematical neuron model (1943). Stateless binary threshold:
y = 1 if Σ(w_i · x_i) ≥ θ, else 0.

No state variables, no dynamics, no reset. Pure combinational logic.
Implements any linearly separable Boolean function. Can compose
AND, OR, NOT gates. Fastest model in zoo: ~2.3M steps/s.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.mcculloch_pitts import McCullochPittsNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: McCullochPittsNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION — defaults, binary output, stateless, reset
# ---------------------------------------------------------------------------
class TestMPIsolation:
    def test_defaults(self):
        n = McCullochPittsNeuron()
        assert n.theta == 1.0

    def test_step_returns_binary(self):
        assert McCullochPittsNeuron().step(0.0) in (0, 1)

    def test_stateless(self):
        """Same input always gives same output regardless of history."""
        n = McCullochPittsNeuron()
        n.step(5.0)
        n.step(5.0)
        assert n.step(0.5) == 0
        assert n.step(5.0) == 1

    def test_no_state_variables(self):
        """Only has theta (parameter), no evolving state."""
        n = McCullochPittsNeuron()
        n.step(5.0)
        n.step(0.0)
        # No v, no w, no gating — only theta
        assert not hasattr(n, "v")

    def test_reset_noop(self):
        n = McCullochPittsNeuron()
        n.step(5.0)
        n.reset()
        assert n.theta == 1.0

    def test_deterministic(self):
        n1 = McCullochPittsNeuron()
        n2 = McCullochPittsNeuron()
        for i in range(200):
            x = float(i) / 100.0 - 0.5
            assert n1.step(x) == n2.step(x)


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — threshold comparison, boundary, transfer function
# ---------------------------------------------------------------------------
class TestMPAnalytical:
    def test_below_threshold(self):
        n = McCullochPittsNeuron()
        assert n.step(0.5) == 0
        assert n.step(0.999) == 0

    def test_at_threshold(self):
        """At x == θ: fires (≥ condition)."""
        n = McCullochPittsNeuron()
        assert n.step(1.0) == 1

    def test_above_threshold(self):
        n = McCullochPittsNeuron()
        assert n.step(5.0) == 1

    def test_negative_input(self):
        assert McCullochPittsNeuron().step(-1.0) == 0

    def test_boundary_precision(self):
        """Float boundary: theta - epsilon → 0, theta → 1."""
        n = McCullochPittsNeuron()
        eps = 1e-15
        assert n.step(n.theta - eps) == 0
        assert n.step(n.theta) == 1

    def test_transfer_function_is_heaviside(self):
        """y(x) = H(x - θ): Heaviside step function."""
        n = McCullochPittsNeuron()
        for x in np.linspace(-2.0, 3.0, 100):
            expected = 1 if x >= n.theta else 0
            assert n.step(float(x)) == expected

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, 2.0, 10.0])
    def test_custom_theta(self, theta: float):
        n = McCullochPittsNeuron(theta=theta)
        assert n.step(theta) == 1
        if theta > 0:
            assert n.step(theta - 0.001) == 0

    def test_zero_theta_fires_at_zero(self):
        """θ=0 → fires at x≥0."""
        n = McCullochPittsNeuron(theta=0.0)
        assert n.step(0.0) == 1
        assert n.step(-0.001) == 0

    def test_negative_theta(self):
        """θ<0 → fires at negative inputs too."""
        n = McCullochPittsNeuron(theta=-1.0)
        assert n.step(-0.5) == 1
        assert n.step(-2.0) == 0


# ---------------------------------------------------------------------------
# 3. LOGIC GATES — universal computation basis
# ---------------------------------------------------------------------------
class TestMPLogicGates:
    def test_and_gate(self):
        """AND: θ=2, inputs ∈ {0,1}."""
        n = McCullochPittsNeuron(theta=2.0)
        assert n.step(0.0 + 0.0) == 0
        assert n.step(1.0 + 0.0) == 0
        assert n.step(0.0 + 1.0) == 0
        assert n.step(1.0 + 1.0) == 1

    def test_or_gate(self):
        """OR: θ=1, inputs ∈ {0,1}."""
        n = McCullochPittsNeuron(theta=1.0)
        assert n.step(0.0 + 0.0) == 0
        assert n.step(1.0 + 0.0) == 1
        assert n.step(0.0 + 1.0) == 1
        assert n.step(1.0 + 1.0) == 1

    def test_not_gate(self):
        """NOT: θ=0, input negated (w=-1). y = 1 if -x ≥ 0."""
        n = McCullochPittsNeuron(theta=0.0)
        # NOT(1): weighted_input = -1 → -1 < 0 → 0
        assert n.step(-1.0) == 0
        # NOT(0): weighted_input = 0 → 0 ≥ 0 → 1
        assert n.step(0.0) == 1

    def test_nand_gate(self):
        """NAND: θ=-1 with weights w1=w2=-1. Input = -x1 - x2."""
        n = McCullochPittsNeuron(theta=-1.0)
        assert n.step(-0.0 - 0.0) == 1  # NAND(0,0) = 1
        assert n.step(-1.0 - 0.0) == 1  # NAND(1,0) = 1
        assert n.step(-0.0 - 1.0) == 1  # NAND(0,1) = 1
        assert n.step(-1.0 - 1.0) == 0  # NAND(1,1) = 0

    def test_three_input_majority(self):
        """Majority gate (3 inputs): θ=2."""
        n = McCullochPittsNeuron(theta=2.0)
        assert n.step(0.0) == 0  # 0+0+0
        assert n.step(1.0) == 0  # 1+0+0
        assert n.step(2.0) == 1  # 1+1+0
        assert n.step(3.0) == 1  # 1+1+1

    def test_linear_separability(self):
        """MP can represent any linearly separable function."""
        # XOR is NOT linearly separable → single MP cannot compute it
        n = McCullochPittsNeuron(theta=1.0)
        xor_inputs = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]
        # No single theta can separate all 4 correctly
        results = [n.step(float(a) + float(b)) for a, b, _ in xor_inputs]
        expected_xor = [e for _, _, e in xor_inputs]
        assert results != expected_xor  # Cannot implement XOR


class TestMPValidation:
    @pytest.mark.parametrize("theta", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_threshold(self, theta: float):
        with pytest.raises(ValueError, match="theta"):
            McCullochPittsNeuron(theta=theta)

    @pytest.mark.parametrize("weighted_input", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_weighted_input(self, weighted_input: float):
        with pytest.raises(ValueError, match="weighted_input"):
            McCullochPittsNeuron().step(weighted_input)

    @pytest.mark.parametrize("theta", [np.nan, np.inf, -np.inf])
    def test_rejects_corrupted_runtime_threshold_before_comparison(self, theta: float):
        n = McCullochPittsNeuron(theta=1.0)
        n.theta = theta
        with pytest.raises(ValueError, match="theta"):
            n.step(2.0)

    def test_runtime_threshold_comparison_matches_heaviside_boundary_after_mutation(self):
        n = McCullochPittsNeuron(theta=1.0)
        n.theta = 2.0
        assert n.step(1.999999999999999) == 0
        assert n.step(2.0) == 1


# ---------------------------------------------------------------------------
# 4. DYNAMICS — firing rate under constant/varying input
# ---------------------------------------------------------------------------
class TestMPDynamics:
    def test_fires_every_step_above_threshold(self):
        """Stateless → fires every step if input ≥ θ."""
        n = McCullochPittsNeuron()
        assert all(n.step(2.0) == 1 for _ in range(1000))

    def test_never_fires_below_threshold(self):
        n = McCullochPittsNeuron()
        assert all(n.step(0.5) == 0 for _ in range(1000))

    def test_rate_is_binary(self):
        """Rate is either 0% or 100% — no intermediate rates."""
        n = McCullochPittsNeuron()
        train_above = [n.step(2.0) for _ in range(1000)]
        train_below = [n.step(0.5) for _ in range(1000)]
        assert sum(train_above) == 1000
        assert sum(train_below) == 0


# ---------------------------------------------------------------------------
# 5. PERFORMANCE — fastest model in zoo
# ---------------------------------------------------------------------------
class TestMPPerformance:
    def test_isolation_throughput(self):
        n = McCullochPittsNeuron()
        N = 1_000_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # Pure comparison — should be ~2M+ steps/s
        assert rate > 500_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(McCullochPittsNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 50 * 500
        rate = neuron_steps / elapsed
        assert rate > 5_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE — Population, Projection, Network, Analysis
# ---------------------------------------------------------------------------
class TestMPPipeline:
    def test_population(self):
        assert Population(McCullochPittsNeuron, n=10, label="mp").n == 10

    def test_projection_wiring(self):
        src = Population(McCullochPittsNeuron, n=5, label="src")
        tgt = Population(McCullochPittsNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(McCullochPittsNeuron, n=10, label="mp")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = McCullochPittsNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc == 5000  # fires every step above threshold

    def test_analysis_firing_rate(self):
        n = McCullochPittsNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_zero_below_threshold(self):
        n = McCullochPittsNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(5000)])
        assert spike_count(train) == 0
