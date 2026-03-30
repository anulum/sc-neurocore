# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SigmaDeltaNeuron

"""Full pipeline test for SigmaDeltaNeuron (Yoon 2017).

Event-driven sigma-delta encoding. Accumulates input in sigma, fires +1
when sigma ≥ θ, fires -1 when sigma ≤ -θ. Subtract-on-spike (not reset).
Ternary output {-1, 0, +1}. Signal reconstruction error bounded by θ."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.sigma_delta import SigmaDeltaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


# ---------------------------------------------------------------------------
# 1. Isolation — construction, ternary output, accumulator mechanics
# ---------------------------------------------------------------------------


class TestSigmaDeltaIsolation:
    def test_construction_defaults(self):
        n = SigmaDeltaNeuron()
        assert n.sigma == 0.0
        assert n.v_threshold == 1.0

    def test_step_returns_ternary(self):
        """Output is {-1, 0, +1}, not just binary."""
        n = SigmaDeltaNeuron()
        assert n.step(0.0) == 0
        n2 = SigmaDeltaNeuron()
        assert n2.step(1.5) == 1  # sigma=1.5 ≥ 1.0 → +1
        n3 = SigmaDeltaNeuron()
        assert n3.step(-1.5) == -1  # sigma=-1.5 ≤ -1.0 → -1

    def test_sigma_accumulates(self):
        """Input is summed into sigma (integration)."""
        n = SigmaDeltaNeuron(v_threshold=100.0)  # high threshold to avoid spikes
        for _ in range(10):
            n.step(0.3)
        assert abs(n.sigma - 3.0) < 1e-10

    def test_reset(self):
        n = SigmaDeltaNeuron()
        for _ in range(50):
            n.step(0.5)
        n.reset()
        assert n.sigma == 0.0


# ---------------------------------------------------------------------------
# 2. Subtract-on-spike mechanism (not reset-to-zero)
# ---------------------------------------------------------------------------


class TestSigmaDeltaSubtract:
    def test_positive_spike_subtracts_threshold(self):
        """On +1 spike: sigma -= threshold (not reset to 0)."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        # sigma = 0 + 1.3 = 1.3 ≥ 1.0 → spike, sigma = 1.3 - 1.0 = 0.3
        s = n.step(1.3)
        assert s == 1
        assert abs(n.sigma - 0.3) < 1e-10

    def test_negative_spike_adds_threshold(self):
        """On -1 spike: sigma += threshold."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        s = n.step(-1.3)
        assert s == -1
        assert abs(n.sigma - (-0.3)) < 1e-10

    def test_residual_carries_over(self):
        """Residual after subtraction carries into next step."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        n.step(0.7)  # sigma = 0.7, no spike
        n.step(0.7)  # sigma = 1.4 ≥ 1.0 → spike, sigma = 0.4
        assert abs(n.sigma - 0.4) < 1e-10

    def test_overflow_accumulation(self):
        """When I > threshold, sigma grows because only one threshold
        is subtracted per step (even if sigma >> threshold)."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        for _ in range(100):
            n.step(2.0)  # Each step: sigma += 2.0, then sigma -= 1.0 → net +1.0
        # After 100 steps: sigma should be large (100 * 1.0 = 100)
        assert n.sigma > 50


# ---------------------------------------------------------------------------
# 3. Signal encoding — rate coding and reconstruction
# ---------------------------------------------------------------------------


class TestSigmaDeltaEncoding:
    def test_spike_rate_equals_input_over_threshold(self):
        """For constant I ∈ (0, θ): rate = I/θ spikes per step."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        I = 0.3
        outputs = [n.step(I) for _ in range(10000)]
        pos = outputs.count(1)
        expected = 10000 * I / 1.0
        assert abs(pos - expected) <= 2, f"pos={pos}, expected={expected}"

    @pytest.mark.parametrize("I", [0.1, 0.25, 0.5, 0.75])
    def test_rate_proportional_to_input(self, I: float):
        """Rate = I/θ for I ∈ (0, θ) — exact for sigma-delta."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        outputs = [n.step(I) for _ in range(10000)]
        pos = outputs.count(1)
        expected = 10000 * I
        assert abs(pos - expected) <= 2

    def test_negative_input_produces_negative_spikes(self):
        n = SigmaDeltaNeuron()
        outputs = [n.step(-0.5) for _ in range(1000)]
        assert outputs.count(-1) == 500
        assert outputs.count(1) == 0

    def test_signal_reconstruction_bounded(self):
        """Cumulative output × θ tracks cumulative input within ±θ.

        This is the fundamental sigma-delta guarantee: the quantisation
        error (sigma residual) is always bounded by the threshold.
        """
        n = SigmaDeltaNeuron(v_threshold=1.0)
        I_signal = np.sin(np.arange(1000) * 0.05) * 0.4
        outputs = np.array([n.step(float(x)) for x in I_signal])
        cumsum_in = np.cumsum(I_signal)
        cumsum_out = np.cumsum(outputs) * n.v_threshold
        max_error = np.max(np.abs(cumsum_in - cumsum_out))
        assert max_error < n.v_threshold + 0.01, (
            f"Reconstruction error {max_error:.4f} exceeds threshold {n.v_threshold}"
        )

    def test_dc_removal(self):
        """For I=0, no spikes ever — perfect silence."""
        n = SigmaDeltaNeuron()
        outputs = [n.step(0.0) for _ in range(10000)]
        assert all(o == 0 for o in outputs)

    def test_bidirectional_encoding(self):
        """Alternating +/- input produces both +1 and -1 spikes."""
        n = SigmaDeltaNeuron(v_threshold=0.5)
        outputs = []
        for t in range(1000):
            I = 0.6 if t % 2 == 0 else -0.6
            outputs.append(n.step(I))
        assert 1 in outputs and -1 in outputs


# ---------------------------------------------------------------------------
# 4. Threshold parameter
# ---------------------------------------------------------------------------


class TestSigmaDeltaThreshold:
    def test_lower_threshold_higher_rate(self):
        n_low = SigmaDeltaNeuron(v_threshold=0.5)
        n_high = SigmaDeltaNeuron(v_threshold=2.0)
        s_low = sum(1 for _ in range(1000) if n_low.step(0.3) == 1)
        s_high = sum(1 for _ in range(1000) if n_high.step(0.3) == 1)
        assert s_low > s_high

    def test_threshold_controls_quantisation_step(self):
        """Each spike represents ±θ of accumulated signal."""
        n = SigmaDeltaNeuron(v_threshold=2.0)
        # I=0.5 → rate = 0.5/2.0 = 0.25 spikes/step
        outputs = [n.step(0.5) for _ in range(10000)]
        pos = outputs.count(1)
        expected = 10000 * 0.5 / 2.0
        assert abs(pos - expected) <= 2

    def test_very_small_threshold(self):
        """θ → 0: almost every step produces a spike."""
        n = SigmaDeltaNeuron(v_threshold=0.01)
        outputs = [n.step(0.1) for _ in range(100)]
        pos = outputs.count(1)
        assert pos >= 90  # rate = 0.1/0.01 = 10, but max 1/step → 100


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestSigmaDeltaEdgeCases:
    def test_exact_threshold_crossing(self):
        """sigma exactly equals threshold → spike."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        s = n.step(1.0)
        assert s == 1
        assert abs(n.sigma) < 1e-10  # 1.0 - 1.0 = 0.0

    def test_exact_negative_threshold(self):
        n = SigmaDeltaNeuron(v_threshold=1.0)
        s = n.step(-1.0)
        assert s == -1
        assert abs(n.sigma) < 1e-10

    def test_large_input_single_spike(self):
        """Even at I=100, only one spike per step (no multi-spike)."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        s = n.step(100.0)
        assert s == 1  # only one +1, not 100

    def test_state_finite_long_run(self):
        """With I > θ, sigma grows unboundedly but stays finite."""
        n = SigmaDeltaNeuron()
        for _ in range(100000):
            n.step(2.0)
        assert np.isfinite(n.sigma)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SigmaDeltaNeuron()
            trace = [(n.step(0.3), n.sigma) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 6. Network pipeline wiring
# ---------------------------------------------------------------------------


class TestSigmaDeltaNetwork:
    def test_population(self):
        pop = Population(SigmaDeltaNeuron, n=10, label="sd")
        assert pop.n == 10

    def test_network_with_drive(self):
        pop = Population(SigmaDeltaNeuron, n=10, label="sd")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


# ---------------------------------------------------------------------------
# 7. Analysis pipeline
# ---------------------------------------------------------------------------


class TestSigmaDeltaAnalysis:
    def test_spike_count(self):
        n = SigmaDeltaNeuron()
        train = np.array([float(max(0, n.step(0.3))) for _ in range(10000)])
        assert spike_count(train) > 100

    def test_spike_count_consistency(self):
        n = SigmaDeltaNeuron()
        train = np.array([float(max(0, n.step(0.3))) for _ in range(10000)])
        assert spike_count(train) == int(train.sum())
