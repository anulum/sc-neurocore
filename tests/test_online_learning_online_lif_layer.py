# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOnlineLIFLayer from former test_online_learning.py

"""Focused suite: TestOnlineLIFLayer from former test_online_learning.py."""

from __future__ import annotations

from tests.online_learning_support import *  # noqa: F403

class TestOnlineLIFLayer:
    def test_step(self):
        layer = OnlineLIFLayer(n_inputs=4, n_neurons=8)
        x = np.random.rand(4)
        spikes = layer.step(x)
        assert spikes.shape == (8,)
        assert set(np.unique(spikes)).issubset({0.0, 1.0})

    def test_reset(self):
        layer = OnlineLIFLayer(n_inputs=4, n_neurons=8)
        layer.step(np.ones(4))
        layer.reset()
        assert np.allclose(layer._v, 0)

    def test_apply_learning_signal(self):
        layer = OnlineLIFLayer(n_inputs=4, n_neurons=8, lr=0.1)
        layer.step(np.ones(4))
        w_before = layer.W.copy()
        layer.apply_learning_signal(np.ones(8))
        assert not np.allclose(layer.W, w_before)
