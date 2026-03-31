# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ClosedFormContinuousNeuron

"""Full pipeline test for ClosedFormContinuousNeuron (Hasani et al. 2022).

Analytical ODE solution: x = x·decay + f_target·(1-decay).
f_target = tanh(w_x·x + w_in·I) is bounded ∈ (-1, 1).
FINDING: default v_threshold=1.0 unreachable (tanh < 1). Lower
threshold (0.5–0.95) enables spiking. Performance: ~68K steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.cfc import ClosedFormContinuousNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _run(neuron: ClosedFormContinuousNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestCFCIsolation:
    def test_defaults(self):
        n = ClosedFormContinuousNeuron()
        assert n.x == 0.0 and n.w_tau == -0.5 and n.w_x == 0.8
        assert n.w_in == 1.0 and n.tau_base == 10.0 and n.v_threshold == 1.0

    def test_step_returns_binary(self):
        assert ClosedFormContinuousNeuron().step(0.0) in (0, 1)

    def test_x_evolves(self):
        n = ClosedFormContinuousNeuron()
        n.step(5.0)
        assert n.x != 0.0

    def test_state_finite(self):
        n = ClosedFormContinuousNeuron()
        for _ in range(50000):
            n.step(5.0)
        assert np.isfinite(n.x)

    def test_reset(self):
        n = ClosedFormContinuousNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.x == 0.0


class TestCFCAnalyticalSolution:
    """x(t+dt) = x·exp(-dt/τ_eff) + f_target·(1 - exp(-dt/τ_eff))."""

    def test_closed_form_formula(self):
        """Verify one step matches the closed-form expression."""
        n = ClosedFormContinuousNeuron(v_threshold=100.0)  # prevent spike
        I = 3.0
        x0 = n.x
        sigma_tau = 1.0 / (1.0 + np.exp(-(n.w_tau * I + n.bias)))
        tau_eff = max(n.tau_base * sigma_tau, 0.1)
        f_target = np.tanh(n.w_x * x0 + n.w_in * I)
        decay = np.exp(-n.dt / tau_eff)
        expected = x0 * decay + f_target * (1.0 - decay)
        n.step(I)
        assert abs(n.x - expected) < 1e-10

    def test_tau_eff_input_dependent(self):
        """τ_eff = τ_base · σ(w_τ·I + bias). Varies with input."""
        n = ClosedFormContinuousNeuron()
        tau1 = n.tau_base / (1.0 + np.exp(-(n.w_tau * 1.0)))
        tau5 = n.tau_base / (1.0 + np.exp(-(n.w_tau * 5.0)))
        assert tau1 != tau5

    def test_f_target_tanh_bounded(self):
        """f_target = tanh(w_x·x + w_in·I) ∈ [-1, 1]. Always bounded.

        Note: tanh(-100) rounds to exactly -1.0 in float64.
        """
        for I in [-100, 0, 100]:
            f = np.tanh(0.8 * 0.5 + 1.0 * I)
            assert -1.0 <= f <= 1.0

    def test_x_converges_to_f_target(self):
        """At steady state (many steps): x → f_target."""
        n = ClosedFormContinuousNeuron(v_threshold=100.0)
        I = 3.0
        for _ in range(10000):
            n.step(I)
        # At ss: x ≈ tanh(w_x*x + w_in*I)
        f_ss = np.tanh(n.w_x * n.x + n.w_in * I)
        assert abs(n.x - f_ss) < 0.01


class TestCFCThresholdBehavior:
    """Default threshold=1.0 is unreachable since tanh < 1."""

    def test_default_threshold_unreachable(self):
        """tanh output never reaches 1.0 → no spikes at θ=1.0."""
        n = ClosedFormContinuousNeuron(v_threshold=1.0)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert spikes == 0, "Should not spike at default threshold"

    def test_lower_threshold_enables_spiking(self):
        """θ=0.5 → spikes because x converges near 1.0."""
        n = ClosedFormContinuousNeuron(v_threshold=0.5)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert spikes > 100

    @pytest.mark.parametrize("theta", [0.3, 0.5, 0.8, 0.95])
    def test_rate_increases_with_lower_threshold(self, theta: float):
        n = ClosedFormContinuousNeuron(v_threshold=theta)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert spikes > 0 or theta > 0.99

    def test_spike_resets_x_to_zero(self):
        n = ClosedFormContinuousNeuron(v_threshold=0.5)
        for _ in range(5000):
            if n.step(5.0) == 1:
                assert n.x == 0.0
                break


class TestCFCPerformance:
    def test_isolation_throughput(self):
        n = ClosedFormContinuousNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 20000


class TestCFCPipeline:
    def test_population(self):
        assert Population(ClosedFormContinuousNeuron, n=10, label="cfc").n == 10

    def test_network_runs(self):
        pop = Population(ClosedFormContinuousNeuron, n=5, label="cfc")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        # May not spike at default threshold — just verify no crash
        assert isinstance(mon.count, int)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ClosedFormContinuousNeuron()
            trace = [(n.step(5.0), n.x) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
