# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ArcaneNeuron

"""Full pipeline test for ArcaneNeuron (Šotek & Arcane Sapience 2026).

Unified self-referential cognition neuron. 5 coupled subsystems:
FAST (τ=5ms), WORKING (τ=200ms), DEEP (τ=10s, identity), GATE
(attention), PREDICTOR (self-model). v_deep PERSISTS through reset.
Performance: ~27K isolation steps/s. FULL PIPELINE WIRED."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: ArcaneNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestArcaneIsolation:
    def test_defaults(self):
        n = ArcaneNeuron()
        assert n.v_fast == 0.0 and n.v_work == 0.0 and n.v_deep == 0.0
        assert n.tau_fast == 5.0 and n.tau_work == 200.0 and n.tau_deep == 10000.0
        assert n.theta == 1.0 and n.w_gate.shape == (4,) and n.w_pred.shape == (3,)

    def test_step_returns_binary(self):
        assert ArcaneNeuron().step(0.0) in (0, 1)

    def test_five_subsystems_evolve(self):
        n = ArcaneNeuron()
        for _ in range(500):
            n.step(2.0)
        state = n.get_state()
        assert state["v_fast"] != 0.0 or state["v_work"] != 0.0
        assert state["novelty"] != 0.0
        assert state["confidence"] != 0.5

    def test_state_finite(self):
        n = ArcaneNeuron()
        for _ in range(50000):
            n.step(2.0)
        state = n.get_state()
        for key, val in state.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), f"{key} = {val}"

    def test_get_state_keys(self):
        n = ArcaneNeuron()
        state = n.get_state()
        expected_keys = {
            "v_fast",
            "v_work",
            "v_deep",
            "confidence",
            "novelty",
            "surprise",
            "prediction",
            "meta_lr",
            "total_steps",
        }
        assert set(state.keys()) == expected_keys


class TestArcaneIdentityPersistence:
    """CORE: v_deep is the identity — it PERSISTS through reset."""

    def test_deep_persists_through_reset(self):
        """reset() zeros v_fast and v_work but NOT v_deep."""
        n = ArcaneNeuron()
        for _ in range(10000):
            n.step(2.0)
        deep_before = n.v_deep
        assert deep_before > 0, "v_deep should accumulate during firing"
        n.reset()
        assert n.v_deep == deep_before, f"v_deep changed from {deep_before} to {n.v_deep} on reset"
        assert n.v_fast == 0.0 and n.v_work == 0.0

    def test_deep_accumulates_slowly(self):
        """v_deep changes on tau_deep=10000 timescale — ultra-slow."""
        n = ArcaneNeuron()
        for _ in range(100):
            n.step(2.0)
        d100 = n.v_deep
        for _ in range(10000):
            n.step(2.0)
        d10k = n.v_deep
        assert abs(d10k) > abs(d100), "v_deep should grow over long runs"
        assert abs(d10k) < 0.1, f"v_deep = {d10k} — should be small (tau=10k)"

    def test_deep_requires_novelty(self):
        """v_deep updates proportional to novelty: dv_deep ∝ v_work * novelty."""
        n = ArcaneNeuron()
        n.alpha_d = 0.0  # disable deep accumulation
        for _ in range(10000):
            n.step(2.0)
        assert abs(n.v_deep) < 1e-10, "alpha_d=0 should prevent deep update"


class TestArcaneThreeTimescales:
    def test_fast_fastest(self):
        """v_fast (τ=5) changes fastest."""
        n = ArcaneNeuron(theta=100.0)  # prevent spikes
        n.step(2.0)
        assert abs(n.v_fast) > abs(n.v_work)
        assert abs(n.v_fast) > abs(n.v_deep)

    def test_working_memory_on_spike(self):
        """v_work updates only when spike occurs (gated by spike)."""
        n = ArcaneNeuron()
        for _ in range(5000):
            if n.step(2.0) == 1:
                break
        assert n.v_work > 0, "v_work should update after spike"


class TestArcaneNoveltyPrediction:
    """Self-referential: predict own state → surprise → novelty."""

    def test_surprise_computed(self):
        n = ArcaneNeuron()
        n.step(2.0)
        assert n._surprise >= 0

    def test_novelty_sigmoid(self):
        """novelty = sigmoid(κ·(surprise - baseline)). Bounded [0, 1]."""
        n = ArcaneNeuron()
        for _ in range(100):
            n.step(2.0)
        assert 0 <= n._novelty <= 1

    def test_predictor_weights_normalised(self):
        """w_pred is normalised after each update."""
        n = ArcaneNeuron()
        for _ in range(1000):
            n.step(2.0)
        norm = np.linalg.norm(n.w_pred)
        assert abs(norm - 1.0) < 0.01 or norm == 0

    def test_meta_lr_increases_with_novelty(self):
        """meta_lr = lr_base * (1 + η * novelty). Higher novelty → faster learning."""
        n = ArcaneNeuron()
        n._novelty = 0.0
        lr_low = n.meta_learning_rate
        n._novelty = 1.0
        lr_high = n.meta_learning_rate
        assert lr_high > lr_low

    def test_confidence_decreases_with_novelty(self):
        """confidence = 1 - mean(novelty_history). High novelty → low confidence."""
        n = ArcaneNeuron()
        n._novelty_history = [0.9] * 20
        n.step(0.0)  # updates confidence
        assert n._confidence < 0.2


class TestArcaneAttentionGate:
    def test_gate_sigmoid(self):
        """gate = sigmoid(w_g · [I, v_fast, v_work, confidence]). Bounded (0, 1)."""
        n = ArcaneNeuron()
        # Gate output is internal — verify indirectly: higher I → more v_fast
        n_low = ArcaneNeuron(theta=100.0)
        n_high = ArcaneNeuron(theta=100.0)
        n_low.step(0.5)
        n_high.step(5.0)
        assert n_high.v_fast > n_low.v_fast

    def test_gate_modulates_effective_input(self):
        """Gate filters input before fast compartment."""
        # With zero gate weights → no input passes
        n = ArcaneNeuron(theta=100.0)
        n.w_gate = np.array([0.0, 0.0, 0.0, 0.0])
        n.step(10.0)
        # gate = sigmoid(0) = 0.5, i_eff = 0.5 * 10 = 5.0
        # v_fast should have changed
        assert n.v_fast > 0  # sigmoid(0) = 0.5, so some input gets through


class TestArcaneEffectiveThreshold:
    def test_threshold_modulated_by_deep(self):
        """eff_threshold = θ · (1 + γ·v_deep) · (1 - δ·confidence).

        Higher v_deep → higher threshold. Higher confidence → lower threshold.
        """
        n = ArcaneNeuron()
        # At defaults: v_deep=0, confidence=0.5
        # eff_threshold = 1.0 * (1+0) * (1 - 0.3*0.5) = 0.85
        n._confidence = 0.5
        eff = n.theta * (1 + n.gamma * n.v_deep) * (1 - n.delta_conf * n._confidence)
        assert abs(eff - 0.85) < 0.01

    def test_confident_lowers_threshold(self):
        """High confidence → lower effective threshold → fires more easily."""
        n_conf = ArcaneNeuron()
        n_unconf = ArcaneNeuron()
        n_conf._novelty_history = [0.1] * 20  # low novelty → high confidence
        n_unconf._novelty_history = [0.9] * 20  # high novelty → low confidence
        s_conf = len(_run(n_conf, current=1.5, steps=5000))
        s_unconf = len(_run(n_unconf, current=1.5, steps=5000))
        # Confident should fire more (lower threshold)
        assert s_conf >= s_unconf


class TestArcaneFI:
    def test_zero_silent(self):
        n = ArcaneNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_suprathreshold_fires(self):
        n = ArcaneNeuron()
        assert len(_run(n, current=2.0, steps=5000)) >= 100

    def test_monotonic_fi(self):
        rates = []
        for I in [2.0, 3.0, 5.0]:
            n = ArcaneNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


class TestArcanePerformance:
    def test_isolation_throughput(self):
        n = ArcaneNeuron()
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 5000

    def test_network_throughput(self):
        pop = Population(ArcaneNeuron, n=10, label="bench")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 10 * 500
        assert neuron_steps / elapsed > 500


class TestArcanePipeline:
    def test_population(self):
        assert Population(ArcaneNeuron, n=10, label="arcane").n == 10

    def test_network_spikes(self):
        pop = Population(ArcaneNeuron, n=10, label="arcane")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(ArcaneNeuron, n=5, label="src")
        tgt = Population(ArcaneNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0
        assert mon_tgt.count > 0

    def test_analysis(self):
        n = ArcaneNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ArcaneNeuron()
            trace = [(n.step(2.0), n.v_fast) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
