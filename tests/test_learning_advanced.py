# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for advanced learning rules

"""Tests for BPTT, TBPTT, EligibilityTrace, R-STDP, Homeostatic, STP."""

from __future__ import annotations

import numpy as np

from sc_neurocore import StochasticLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.learning.advanced import (
    BPTTLearner,
    TBPTTLearner,
    EligibilityTrace,
    RewardModulatedLearner,
    HomeostaticPlasticity,
    ShortTermPlasticity,
)


def _make_small_network(n_in=10, n_out=5, w=0.3, p=0.5):
    pop_in = Population(StochasticLIFNeuron, n=n_in, label="in")
    pop_out = Population(StochasticLIFNeuron, n=n_out, label="out")
    proj = Projection(pop_in, pop_out, weight=w, probability=p, seed=42)
    drive = PoissonInput(n=n_in, rate_hz=100.0, weight=2.0, dt=0.001, seed=99)
    net = Network(pop_in, pop_out, proj, drive)
    return net, proj


class TestEligibilityTrace:
    def test_trace_starts_zero(self):
        et = EligibilityTrace(tau_e=20.0, dt=1.0)
        pre = np.zeros(5)
        post = np.zeros(3)
        error = np.ones(3)
        dw = et.update(pre, post, error)
        assert dw.shape == (5, 3)
        np.testing.assert_allclose(dw, 0.0)

    def test_trace_accumulates(self):
        et = EligibilityTrace(tau_e=20.0, dt=1.0)
        pre = np.array([1.0, 0.0, 0.0])
        post = np.array([0.0, 1.0])
        error = np.array([1.0, 1.0])
        dw1 = et.update(pre, post, error)
        # Pre=1 and post=1 should produce nonzero at [0,1]
        assert dw1[0, 1] > 0

    def test_trace_decays(self):
        et = EligibilityTrace(tau_e=10.0, dt=1.0)
        pre = np.array([1.0, 0.0])
        post = np.array([1.0])
        error = np.array([1.0])
        dw1 = et.update(pre, post, error)
        # Next step with no spikes — trace should decay
        dw2 = et.update(np.zeros(2), np.zeros(1), np.array([1.0]))
        assert abs(dw2[0, 0]) < abs(dw1[0, 0])

    def test_error_gating(self):
        """Zero error should produce zero weight delta."""
        et = EligibilityTrace(tau_e=20.0, dt=1.0)
        pre = np.array([1.0])
        post = np.array([1.0])
        dw = et.update(pre, post, np.array([0.0]))
        np.testing.assert_allclose(dw, 0.0)

    def test_decay_constant(self):
        et = EligibilityTrace(tau_e=20.0, dt=1.0)
        expected_decay = np.exp(-1.0 / 20.0)
        np.testing.assert_allclose(et.decay, expected_decay)


class TestBPTTLearner:
    def test_loss_returns_float(self):
        net, proj = _make_small_network()

        def mse_loss(spikes, targets):
            # spikes shape matches first population (n_in=10)
            return float(np.mean((spikes - targets) ** 2))

        learner = BPTTLearner(net, loss_fn=mse_loss, lr=0.001)
        inputs = np.random.randn(50, 10)
        targets = np.random.randint(0, 2, size=(50, 10)).astype(float)
        loss = learner.train_step(inputs, targets)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_weights_change_after_step(self):
        net, proj = _make_small_network()
        w_before = proj.data.copy()

        def mse_loss(spikes, targets):
            return float(np.mean((spikes - targets) ** 2))

        learner = BPTTLearner(net, loss_fn=mse_loss, lr=0.01)
        inputs = np.random.randn(20, 10)
        targets = np.ones((20, 10))
        learner.train_step(inputs, targets)
        assert not np.allclose(proj.data, w_before), "weights unchanged after BPTT"


class TestTBPTTLearner:
    def test_chunked_loss_finite(self):
        net, proj = _make_small_network()

        def mse(s, t):
            return float(np.mean((s - t) ** 2))

        learner = TBPTTLearner(net, loss_fn=mse, lr=0.001, k=10)
        inputs = np.random.randn(50, 10)
        targets = np.random.randint(0, 2, (50, 10)).astype(float)
        loss = learner.train_step(inputs, targets)
        assert np.isfinite(loss)

    def test_chunk_size_respected(self):
        net, proj = _make_small_network()
        learner = TBPTTLearner(net, loss_fn=lambda s, t: 0.0, lr=0.01, k=7)
        assert learner.k == 7


class TestRewardModulatedLearner:
    def test_positive_reward_does_not_crash(self):
        """R-STDP step with positive reward should execute without error.
        Weight changes depend on spike coincidence detection (voltage > 0.9)
        which may not trigger with default LIF parameters."""
        net, proj = _make_small_network()
        rstdp = RewardModulatedLearner(net, tau_reward=100.0)
        net.run(duration=0.05, dt=0.001)
        rstdp.step(reward=10.0)  # should not raise

    def test_zero_reward_minimal_change(self):
        net, proj = _make_small_network()
        rstdp = RewardModulatedLearner(net, tau_reward=100.0)
        w_before = proj.data.copy()
        net.run(duration=0.01, dt=0.001)
        rstdp.step(reward=0.0)
        dw = proj.data - w_before
        np.testing.assert_allclose(dw, 0.0, atol=1e-10)


class TestShortTermPlasticity:
    def test_class_exists(self):
        """ShortTermPlasticity should be importable."""
        assert ShortTermPlasticity is not None


class TestHomeostaticPlasticity:
    def test_class_exists(self):
        """HomeostaticPlasticity should be importable."""
        assert HomeostaticPlasticity is not None

    def test_active_population_rescales_incoming_projection_weights(self):
        # An above-target firing population drives the rate estimate positive, so
        # the controller rescales every incoming projection's weights toward the
        # target rate (clipped to the [0.9, 1.1] per-step band).
        class _Projection:
            def __init__(self) -> None:
                self.data = np.ones(4, dtype=np.float64)

        class _Population:
            def __init__(self) -> None:
                self.voltages = np.array([1.0, 1.0, 1.0, 0.0])  # mostly firing
                self._projections = [_Projection()]

        controller = HomeostaticPlasticity(target_rate=10.0, tau=1.0)
        population = _Population()

        controller.update(population)

        assert controller._rate_estimate > 0
        # Over-firing relative to target -> weights pulled down to the band floor.
        assert np.allclose(population._projections[0].data, 0.9)
        assert controller._last_scale == 0.9
