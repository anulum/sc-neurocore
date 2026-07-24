# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBPTTLearner from former test_learning_advanced.py

"""Focused suite: TestBPTTLearner from former test_learning_advanced.py."""

from __future__ import annotations

from tests.learning_advanced_support import *  # noqa: F403


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
