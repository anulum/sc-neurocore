# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTBPTTLearner from former test_learning_advanced.py

"""Focused suite: TestTBPTTLearner from former test_learning_advanced.py."""

from __future__ import annotations

from tests.learning_advanced_support import *  # noqa: F403


class TestTBPTTLearner:
    def test_chunked_loss_finite(self):
        net, proj = _make_small_network()

        def mse(s, t):
            return float(np.mean((s - t) ** 2))

        learner = TBPTTLearner(net, loss_fn=mse, lr=0.001, k=10)
        inputs = np.random.randn(50, 10)
        targets = np.random.randint(0, 2, (50, 5)).astype(float)
        loss = learner.train_step(inputs, targets)
        assert np.isfinite(loss)

    def test_chunk_size_respected(self):
        net, proj = _make_small_network()
        learner = TBPTTLearner(net, loss_fn=lambda s, t: 0.0, lr=0.01, k=7)
        assert learner.k == 7
