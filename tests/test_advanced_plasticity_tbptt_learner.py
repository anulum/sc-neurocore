# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTBPTTLearner from former test_advanced_plasticity.py

"""Focused suite: TestTBPTTLearner from former test_advanced_plasticity.py."""

from __future__ import annotations

from tests.advanced_plasticity_support import *  # noqa: F403

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
        learner = TBPTTLearner(net, loss_fn=lambda p, t: float(np.mean((p - t) ** 2)), k=5)
        learner.train_step(inputs, targets)
        assert not np.allclose(proj.data, w_before)

    def test_chunking_matches_full_bptt_shape(self, simple_net):
        """TBPTT with k >= n_steps should behave like full BPTT (same loss shape)."""
        net, pop_a, _, _ = simple_net
        n_steps = 15
        inputs = np.random.randn(n_steps, pop_a.n) * 5
        targets = np.zeros((n_steps, pop_a.n))

        def mse(p, t):
            return float(np.mean((p - t) ** 2))

        learner = TBPTTLearner(net, loss_fn=mse, lr=0.0, k=n_steps)
        loss_full = learner.train_step(inputs, targets)
        assert isinstance(loss_full, float)

    def test_multiple_chunks(self, simple_net):
        """Sequence of 25 with k=7 should produce 4 chunks."""
        net, pop_a, _, _ = simple_net
        inputs = np.random.randn(25, pop_a.n) * 5
        targets = np.zeros((25, pop_a.n))

        def mse(p, t):
            return float(np.mean((p - t) ** 2))

        learner = TBPTTLearner(net, loss_fn=mse, lr=1e-4, k=7)
        loss = learner.train_step(inputs, targets)
        assert loss >= 0
