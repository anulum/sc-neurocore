# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.few_shot (spike-based meta-learning)
from __future__ import annotations
import numpy as np
from sc_neurocore.few_shot import HebbianFewShot, SpikePrototypeNet


class TestHebbianFewShot:
    def test_store_and_query(self):
        fs = HebbianFewShot(n_features=8, n_classes=3)
        fs.store(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64), label=0)
        fs.store(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64), label=1)
        pred = fs.query(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64))
        assert pred == 0

    def test_few_shot_episode(self):
        fs = HebbianFewShot(n_features=4, n_classes=2)
        support_x = [np.array([1, 0, 0, 0.0]), np.array([0, 0, 1, 0.0])]
        support_y = [0, 1]
        query_x = [np.array([0.9, 0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.9, 0.0])]
        preds = fs.few_shot_episode(support_x, support_y, query_x)
        assert len(preds) == 2
        assert preds[0] == 0
        assert preds[1] == 1

    def test_temporal_input(self):
        fs = HebbianFewShot(n_features=4, n_classes=2)
        spike_train = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
        fs.store(spike_train, label=0)
        pred = fs.query(spike_train)
        assert pred == 0

    def test_reset(self):
        fs = HebbianFewShot(n_features=4, n_classes=2)
        fs.store(np.ones(4), 0)
        fs.reset()
        assert np.allclose(fs.memory, 0)

    def test_5shot(self):
        fs = HebbianFewShot(n_features=8, n_classes=3)
        rng = np.random.RandomState(42)
        for _ in range(5):
            fs.store(rng.rand(8) * np.array([1, 1, 0, 0, 0, 0, 0, 0.0]), 0)
            fs.store(rng.rand(8) * np.array([0, 0, 1, 1, 0, 0, 0, 0.0]), 1)
            fs.store(rng.rand(8) * np.array([0, 0, 0, 0, 1, 1, 0, 0.0]), 2)
        pred = fs.query(np.array([0.9, 0.8, 0, 0, 0, 0, 0, 0.0]))
        assert pred == 0


class TestSpikePrototypeNet:
    def test_classify(self):
        pn = SpikePrototypeNet(n_features=4)
        support_x = [np.array([1, 0, 0, 0.0]), np.array([0, 0, 1, 0.0])]
        support_y = [0, 1]
        query_x = [np.array([0.9, 0.1, 0.0, 0.0])]
        preds = pn.classify(support_x, support_y, query_x)
        assert preds[0] == 0

    def test_euclidean(self):
        pn = SpikePrototypeNet(n_features=4, metric="euclidean")
        support_x = [np.array([1, 0, 0, 0.0]), np.array([0, 0, 0, 1.0])]
        support_y = [0, 1]
        preds = pn.classify(support_x, support_y, [np.array([0.8, 0.1, 0.0, 0.0])])
        assert preds[0] == 0

    def test_multi_support(self):
        pn = SpikePrototypeNet(n_features=4)
        support_x = [np.ones(4), np.ones(4), np.zeros(4), np.zeros(4)]
        support_y = [0, 0, 1, 1]
        preds = pn.classify(support_x, support_y, [np.ones(4) * 0.8])
        assert preds[0] == 0
