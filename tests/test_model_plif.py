# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ParametricLIFNeuron (PLIF)

"""Full pipeline test for ParametricLIFNeuron (Fang et al. 2021).

Parametric LIF with learnable decay alpha = sigmoid(a).
V(t+1) = alpha·V(t)·(1-spike(t)) + I(t).
Spike threshold: I_crit = threshold·(1 - alpha)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.plif import ParametricLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


# ---------------------------------------------------------------------------
# 1. Isolation — construction and learnable parameter
# ---------------------------------------------------------------------------


class TestPLIFIsolation:
    def test_construction_defaults(self):
        n = ParametricLIFNeuron()
        assert n.v == 0.0
        assert n.a == 0.0
        assert n.threshold == 1.0
        assert n.dt == 1.0

    def test_step_returns_binary(self):
        assert ParametricLIFNeuron().step(0.0) in (0, 1)

    def test_alpha_is_sigmoid_of_a(self):
        """alpha = 1 / (1 + exp(-a)) — the learnable decay parameter."""
        for a_val in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            n = ParametricLIFNeuron(a=a_val)
            expected = 1.0 / (1.0 + np.exp(-a_val))
            assert abs(n.alpha - expected) < 1e-12, (
                f"a={a_val}: alpha={n.alpha}, expected={expected}"
            )

    def test_alpha_at_zero(self):
        """a=0 → alpha=0.5 (symmetric midpoint)."""
        assert ParametricLIFNeuron(a=0.0).alpha == 0.5

    def test_alpha_monotonic_in_a(self):
        """alpha increases monotonically with a."""
        alphas = [ParametricLIFNeuron(a=a).alpha for a in [-2, -1, 0, 1, 2]]
        assert all(alphas[i] < alphas[i + 1] for i in range(len(alphas) - 1))

    def test_alpha_bounded_0_1(self):
        """Sigmoid output ∈ (0, 1) for moderate a values."""
        for a_val in [-10.0, -5.0, 5.0, 10.0]:
            alpha = ParametricLIFNeuron(a=a_val).alpha
            assert 0.0 < alpha < 1.0, f"a={a_val}: alpha={alpha}"

    def test_alpha_saturates_at_extreme_a(self):
        """At extreme a, sigmoid saturates in float64.

        exp(-100) ≈ 3.7e-44 (not zero), but exp(100) overflows → alpha=1.0.
        """
        assert ParametricLIFNeuron(a=100.0).alpha == 1.0
        assert ParametricLIFNeuron(a=-100.0).alpha < 1e-40

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": np.nan},
            {"v": np.inf},
            {"a": np.nan},
            {"a": np.inf},
            {"threshold": 0.0},
            {"threshold": np.nan},
            {"dt": 0.0},
            {"dt": np.inf},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs):
        with pytest.raises(ValueError):
            ParametricLIFNeuron(**kwargs)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = ParametricLIFNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    @pytest.mark.parametrize(
        "field",
        ["v", "a", "threshold", "dt"],
    )
    def test_rejects_corrupted_runtime_state_before_mutation(self, field: str):
        n = ParametricLIFNeuron(v=0.25)
        before = n.v
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(0.1)
        if field != "v":
            assert n.v == before

    def test_rejects_non_finite_voltage_candidate_before_mutation(self):
        n = ParametricLIFNeuron(v=1.0e308, a=1000.0, threshold=1.7e308)
        before = n.v
        with pytest.raises(ValueError, match="voltage candidate"):
            n.step(1.0e308)
        assert n.v == before

    def test_alpha_is_stable_for_large_negative_parameter(self):
        n = ParametricLIFNeuron(a=-1000.0)
        assert n.alpha == 0.0


# ---------------------------------------------------------------------------
# 2. Voltage dynamics — geometric accumulation
# ---------------------------------------------------------------------------


class TestPLIFDynamics:
    def test_voltage_accumulation(self):
        """V(t) = sum_{k=0}^{t-1} alpha^k · I = I · (1 - alpha^t) / (1 - alpha).

        For alpha=0.5, I=0.5: V(1)=0.5, V(2)=0.75, V(3)=0.875, ...
        """
        n = ParametricLIFNeuron(a=0.0)  # alpha=0.5
        expected = [0.5, 0.75, 0.875]
        for t, exp_v in enumerate(expected):
            n.step(0.5)
            assert abs(n.v - exp_v) < 1e-12, f"t={t + 1}: v={n.v}, expected={exp_v}"

    def test_steady_state_voltage(self):
        """V_ss = I / (1 - alpha) when V_ss < threshold (no spikes)."""
        n = ParametricLIFNeuron(a=-2.0)  # alpha ≈ 0.119
        I = 0.3
        v_ss_analytical = I / (1.0 - n.alpha)
        # Run long enough to converge
        for _ in range(500):
            n.step(I)
        assert abs(n.v - v_ss_analytical) < 1e-6, (
            f"v={n.v:.6f}, expected V_ss={v_ss_analytical:.6f}"
        )

    def test_geometric_convergence_from_zero(self):
        """Voltage approaches V_ss geometrically: error halves each step at alpha=0.5."""
        n = ParametricLIFNeuron(a=0.0)  # alpha=0.5
        I = 0.3  # V_ss = 0.6, below threshold
        errors = []
        for _ in range(10):
            n.step(I)
            errors.append(abs(n.v - 0.6))
        # Each error ≈ alpha × previous error
        for i in range(1, len(errors)):
            if errors[i - 1] > 1e-12:
                ratio = errors[i] / errors[i - 1]
                assert abs(ratio - 0.5) < 0.01, f"Error ratio = {ratio:.4f}, expected ≈0.5"

    def test_no_leak_when_alpha_near_1(self):
        """With alpha ≈ 1, voltage barely decays — nearly a perfect integrator."""
        n = ParametricLIFNeuron(a=10.0)  # alpha ≈ 0.99995
        n.step(0.3)
        v_after = n.v
        n.step(0.0)  # zero input — should decay by factor alpha
        assert n.v > 0.99 * v_after, f"v decayed from {v_after:.6f} to {n.v:.6f}"

    def test_fast_decay_when_alpha_near_0(self):
        """With alpha ≈ 0, voltage decays almost instantly."""
        n = ParametricLIFNeuron(a=-10.0)  # alpha ≈ 0.00005
        n.step(0.3)
        n.step(0.0)  # zero input
        assert n.v < 0.001, f"v = {n.v} — expected near-zero decay"


# ---------------------------------------------------------------------------
# 3. Threshold and spike mechanism
# ---------------------------------------------------------------------------


class TestPLIFThreshold:
    def test_spike_on_updated_voltage(self):
        """Returned spike is based on updated V, not pre-step V.

        Old V triggers reset (line 33), new V determines returned spike (line 35).
        V_old=1.5 → reset → V_new = 0 + I. Spike returned iff V_new ≥ threshold.
        """
        n = ParametricLIFNeuron()
        n.v = 1.5
        # V_old=1.5 ≥ 1.0 → reset; V_new = alpha*1.5*0 + 0.3 = 0.3 < 1.0
        s = n.step(0.3)
        assert s == 0, "V_new = 0.3 < threshold, should not spike"
        assert abs(n.v - 0.3) < 1e-12

        # Now: V_old=1.5 → reset; V_new = 0 + 1.5 = 1.5 ≥ 1.0 → spike
        n2 = ParametricLIFNeuron()
        n2.v = 1.5
        s2 = n2.step(1.5)
        assert s2 == 1, "V_new = 1.5 ≥ threshold, should spike"

    def test_suprathreshold_input_fires_every_step(self):
        """I ≥ threshold → fires every step (V resets to I ≥ threshold)."""
        n = ParametricLIFNeuron()
        # Skip first step (V starts at 0)
        n.step(1.5)  # V=1.5 (no spike on first since V was 0)
        spikes = sum(n.step(1.5) for _ in range(100))
        assert spikes == 100

    def test_exact_threshold_input(self):
        """I = threshold → fires every step (after first)."""
        n = ParametricLIFNeuron()
        n.step(1.0)  # V=0+1.0=1.0, but spike check was V=0 (no spike)
        # Now V=1.0 → spike, V=0+1.0=1.0
        spikes = sum(n.step(1.0) for _ in range(100))
        assert spikes == 100

    def test_critical_current(self):
        """I_crit = threshold · (1 - alpha). Below this, no spikes ever.

        For alpha=0.5, threshold=1.0: I_crit = 0.5.
        """
        alpha = 0.5
        I_crit = 1.0 * (1.0 - alpha)
        # Just below critical
        n_below = ParametricLIFNeuron(a=0.0)
        spikes_below = sum(n_below.step(I_crit - 0.01) for _ in range(1000))
        assert spikes_below == 0, f"{spikes_below} spikes below I_crit"

    def test_reset_is_soft(self):
        """After spike, V = I (not zero) — soft reset via (1-spike) multiplication."""
        n = ParametricLIFNeuron()
        n.v = 2.0  # will spike
        n.step(0.7)
        assert abs(n.v - 0.7) < 1e-12, "Reset should set V = I, not V = 0"


# ---------------------------------------------------------------------------
# 4. Learnable parameter effect on firing rate
# ---------------------------------------------------------------------------


class TestPLIFLearnableRate:
    def test_higher_alpha_more_spikes(self):
        """Higher alpha (more memory) → easier to reach threshold → more spikes."""
        I = 0.4  # Below I_crit for alpha=0.5, above for alpha=0.73
        n_low = ParametricLIFNeuron(a=-1.0)  # alpha ≈ 0.27
        n_high = ParametricLIFNeuron(a=1.0)  # alpha ≈ 0.73
        s_low = sum(n_low.step(I) for _ in range(500))
        s_high = sum(n_high.step(I) for _ in range(500))
        assert s_high > s_low

    @pytest.mark.parametrize("a_val", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_rate_at_suprathreshold(self, a_val: float):
        """At I ≥ threshold, rate = 1 spike/step regardless of alpha."""
        n = ParametricLIFNeuron(a=a_val)
        n.step(2.0)  # prime (V=2.0)
        spikes = sum(n.step(2.0) for _ in range(100))
        assert spikes == 100

    def test_subcritical_no_spikes(self):
        """Below I_crit, neuron never fires (voltage converges below threshold)."""
        for a_val in [-2.0, 0.0, 2.0]:
            n = ParametricLIFNeuron(a=a_val)
            alpha = n.alpha
            I_crit = n.threshold * (1.0 - alpha)
            I_test = I_crit * 0.9  # 10% below critical
            spikes = sum(n.step(I_test) for _ in range(1000))
            assert spikes == 0, (
                f"a={a_val}: {spikes} spikes at I={I_test:.4f} < I_crit={I_crit:.4f}"
            )


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestPLIFEdgeCases:
    def test_zero_input(self):
        """Zero input from rest → V stays at 0, no spikes."""
        n = ParametricLIFNeuron()
        spikes = sum(n.step(0.0) for _ in range(100))
        assert spikes == 0
        assert n.v == 0.0

    def test_negative_input(self):
        """Negative input drives V below 0."""
        n = ParametricLIFNeuron()
        n.step(-0.5)
        assert n.v == -0.5

    def test_reset_method(self):
        n = ParametricLIFNeuron()
        for _ in range(50):
            n.step(2.0)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ParametricLIFNeuron(a=1.0)
            trace = [(n.step(0.5), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 6. Network
# ---------------------------------------------------------------------------


class TestPLIFNetwork:
    def test_population(self):
        pop = Population(ParametricLIFNeuron, n=10, label="plif")
        assert pop.n == 10

    def test_network_spikes(self):
        pop = Population(ParametricLIFNeuron, n=20, label="plif")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=1.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


# ---------------------------------------------------------------------------
# 7. Analysis
# ---------------------------------------------------------------------------


class TestPLIFAnalysis:
    def test_spike_count(self):
        n = ParametricLIFNeuron(a=1.0)
        train = np.array([float(n.step(0.5)) for _ in range(500)])
        assert spike_count(train) > 10

    def test_spike_count_consistency(self):
        n = ParametricLIFNeuron(a=1.0)
        train = np.array([float(n.step(0.5)) for _ in range(500)])
        assert spike_count(train) == int(train.sum())
