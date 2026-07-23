# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBPTTLearner from former test_advanced_plasticity.py

"""Focused suite: TestBPTTLearner from former test_advanced_plasticity.py."""

from __future__ import annotations

from tests.advanced_plasticity_support import *  # noqa: F403

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
