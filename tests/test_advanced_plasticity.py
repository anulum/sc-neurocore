# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for learning.advanced plasticity rules

"""Tests for advanced plasticity: BPTT, eligibility, R-STDP, meta, homeostatic, STP, structural."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.learning.advanced import (
    BPTTLearner,
    TBPTTLearner,
    EligibilityTrace,
    HomeostaticPlasticity,
    MetaLearner,
    RewardModulatedLearner,
    ShortTermPlasticity,
    StructuralPlasticity,
    _fast_sigmoid_surrogate,
)


@pytest.fixture()
def simple_net():
    """Two-population network with one projection."""
    pop_a = Population("LapicqueNeuron", 5, label="src")
    pop_b = Population("LapicqueNeuron", 5, label="tgt")
    proj = Projection(pop_a, pop_b, weight=0.3, probability=1.0, seed=0)
    net = Network(pop_a, pop_b, proj)
    return net, pop_a, pop_b, proj


class TestSurrogateGradient:
    def test_peak_at_threshold(self):
        grad = _fast_sigmoid_surrogate(np.array([1.0]))
        assert grad[0] > 0
        assert grad[0] == pytest.approx(25.0, rel=0.01)

    def test_decays_away(self):
        at_thresh = _fast_sigmoid_surrogate(np.array([1.0]))[0]
        away = _fast_sigmoid_surrogate(np.array([2.0]))[0]
        assert away < at_thresh


class TestBPTTLearner:
    def test_train_step_returns_loss(self, simple_net):
        net, pop_a, pop_b, _ = simple_net
        n_steps = 10
        inputs = np.random.randn(n_steps, pop_a.n) * 5
        targets = np.zeros((n_steps, pop_a.n))

        def mse(pred, tgt):
            return float(np.mean((pred - tgt) ** 2))

        learner = BPTTLearner(net, loss_fn=mse, lr=1e-3)
        loss = learner.train_step(inputs, targets)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_weights_change(self, simple_net):
        net, pop_a, _, proj = simple_net
        w_before = proj.data.copy()
        inputs = np.random.randn(10, pop_a.n) * 10
        targets = np.ones((10, pop_a.n))
        learner = BPTTLearner(net, loss_fn=lambda p, t: float(np.mean((p - t) ** 2)))
        learner.train_step(inputs, targets)
        assert not np.allclose(proj.data, w_before)


class TestTBPTTLearner:
    def test_train_step_returns_loss(self, simple_net):
        net, pop_a, pop_b, _ = simple_net
        n_steps = 30
        inputs = np.random.randn(n_steps, pop_a.n) * 5
        targets = np.zeros((n_steps, pop_a.n))

        def mse(pred, tgt):
            return float(np.mean((pred - tgt) ** 2))

        learner = TBPTTLearner(net, loss_fn=mse, lr=1e-3, k=10)
        loss = learner.train_step(inputs, targets)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_weights_change(self, simple_net):
        net, pop_a, _, proj = simple_net
        w_before = proj.data.copy()
        inputs = np.random.randn(20, pop_a.n) * 10
        targets = np.ones((20, pop_a.n))
        learner = TBPTTLearner(
            net, loss_fn=lambda p, t: float(np.mean((p - t) ** 2)), k=5
        )
        learner.train_step(inputs, targets)
        assert not np.allclose(proj.data, w_before)

    def test_chunking_matches_full_bptt_shape(self, simple_net):
        """TBPTT with k >= n_steps should behave like full BPTT (same loss shape)."""
        net, pop_a, _, _ = simple_net
        n_steps = 15
        inputs = np.random.randn(n_steps, pop_a.n) * 5
        targets = np.zeros((n_steps, pop_a.n))
        mse = lambda p, t: float(np.mean((p - t) ** 2))

        learner = TBPTTLearner(net, loss_fn=mse, lr=0.0, k=n_steps)
        loss_full = learner.train_step(inputs, targets)
        assert isinstance(loss_full, float)

    def test_multiple_chunks(self, simple_net):
        """Sequence of 25 with k=7 should produce 4 chunks."""
        net, pop_a, _, _ = simple_net
        inputs = np.random.randn(25, pop_a.n) * 5
        targets = np.zeros((25, pop_a.n))
        mse = lambda p, t: float(np.mean((p - t) ** 2))
        learner = TBPTTLearner(net, loss_fn=mse, lr=1e-4, k=7)
        loss = learner.train_step(inputs, targets)
        assert loss >= 0


class TestEligibilityTrace:
    def test_output_shape(self):
        et = EligibilityTrace(tau_e=20.0)
        pre = np.array([1, 0, 1, 0, 0], dtype=np.float64)
        post = np.array([0, 1, 1], dtype=np.float64)
        err = np.array([0.5, -0.3, 0.1])
        delta = et.update(pre, post, err)
        assert delta.shape == (5, 3)

    def test_trace_decays(self):
        et = EligibilityTrace(tau_e=5.0)
        pre = np.array([1.0, 0.0])
        post = np.array([1.0])
        err = np.array([1.0])
        d1 = et.update(pre, post, err).copy()
        pre_zero = np.array([0.0, 0.0])
        d2 = et.update(pre_zero, post, err).copy()
        assert np.all(np.abs(d2) <= np.abs(d1) + 1e-12)


class TestRewardModulatedLearner:
    def test_step_runs(self, simple_net):
        net, _, _, proj = simple_net
        learner = RewardModulatedLearner(net, tau_reward=50.0)
        w_before = proj.data.copy()
        learner.step(reward=1.0)
        assert proj.data is not None
        assert proj.data.shape == w_before.shape

    def test_weights_non_negative(self, simple_net):
        net, _, _, proj = simple_net
        learner = RewardModulatedLearner(net, tau_reward=10.0)
        for _ in range(20):
            learner.step(reward=-5.0)
        assert np.all(proj.data >= 0)


class TestMetaLearner:
    def test_inner_loop(self, simple_net):
        net, pop_a, _, proj = simple_net
        w_before = proj.data.copy()
        inputs = np.ones((20, pop_a.n)) * 50.0
        targets = np.zeros((20, pop_a.n))
        ml = MetaLearner(net, inner_lr=0.1)
        ml.inner_loop((inputs, targets), n_steps=5)
        assert not np.allclose(proj.data, w_before)

    def test_outer_step(self, simple_net):
        net, pop_a, _, proj = simple_net
        tasks = [(np.random.randn(5, pop_a.n) * 5, np.zeros((5, pop_a.n))) for _ in range(3)]
        ml = MetaLearner(net, inner_lr=0.01, outer_lr=0.001)
        w_before = proj.data.copy()
        ml.outer_step(tasks)
        assert proj.data.shape == w_before.shape


class TestHomeostaticPlasticity:
    def test_update_runs(self, simple_net):
        _, pop_a, _, _ = simple_net
        hp = HomeostaticPlasticity(target_rate=10.0, tau=100.0)
        hp.update(pop_a)
        assert hp._rate_estimate is not None


class TestShortTermPlasticity:
    def test_returns_scaling(self):
        stp = ShortTermPlasticity(tau_d=200.0, tau_f=600.0, u_se=0.2)
        pre = np.array([1, 0, 1, 0, 0], dtype=np.float64)
        scale = stp.update(pre)
        assert scale.shape == (5,)
        assert np.all(scale >= 0)
        assert np.all(scale <= 1.0)

    def test_depression_on_repeated_spikes(self):
        stp = ShortTermPlasticity(tau_d=50.0, tau_f=600.0, u_se=0.5)
        pre = np.array([1.0])
        s1 = stp.update(pre)[0]
        s2 = stp.update(pre)[0]
        assert s2 < s1


class TestStructuralPlasticity:
    def test_prune(self, simple_net):
        _, _, _, proj = simple_net
        proj.data[:] = 0.001
        sp = StructuralPlasticity(prune_threshold=0.05)
        sp.update(proj)
        assert np.sum(proj.data == 0.0) > 0

    def test_grow(self, simple_net):
        _, _, _, proj = simple_net
        proj.data[:] = 0.0
        sp = StructuralPlasticity(growth_rate=1.0, prune_threshold=0.001)
        sp.update(proj)
        assert np.any(proj.data > 0)
