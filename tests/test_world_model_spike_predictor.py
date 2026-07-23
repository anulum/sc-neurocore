# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikePredictor from former test_world_model.py

"""Focused suite: TestSpikePredictor from former test_world_model.py."""

from __future__ import annotations

from tests.world_model_support import *  # noqa: F403

class TestSpikePredictor:
    def test_construction(self):
        p = SpikePredictor(n_channels=4, history_len=8)
        assert p.W.shape == (4, 32)
        assert p.bias.shape == (4,)

    def test_predict_shape(self):
        p = SpikePredictor(n_channels=4)
        pred = p.predict()
        assert pred.shape == (4,)
        assert pred.dtype == np.int8

    def test_predict_probs_range(self):
        p = SpikePredictor(n_channels=4)
        probs = p.predict_probs()
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_update_changes_weights(self):
        p = SpikePredictor(n_channels=4, lr=0.1)
        # First update populates history (features are zero → W unchanged via outer)
        p.update(np.array([1, 0, 1, 0]))
        # Second update has non-zero history → W changes
        w_before = p.W.copy()
        p.update(np.array([0, 1, 0, 1]))
        assert not np.array_equal(p.W, w_before)

    def test_reset(self):
        p = SpikePredictor(n_channels=4, seed=42)
        w_init = p.W.copy()
        p.update(np.array([1, 0, 1, 0]))
        p.reset()
        np.testing.assert_array_equal(p.W, w_init)

    def test_deterministic(self):
        a = SpikePredictor(n_channels=4, seed=99)
        b = SpikePredictor(n_channels=4, seed=99)
        np.testing.assert_array_equal(a.W, b.W)
        np.testing.assert_array_equal(a.predict(), b.predict())

    def test_learning_improves_prediction(self):
        p = SpikePredictor(n_channels=2, history_len=4, lr=0.05, seed=0)
        # Alternating pattern: [1,0], [0,1], [1,0], [0,1], ...
        pattern = np.array([[1, 0], [0, 1]] * 50)
        errors_first_10 = 0
        errors_last_10 = 0
        for t in range(100):
            pred = p.predict()
            err = int(np.sum(pred != pattern[t]))
            if t < 10:
                errors_first_10 += err
            if t >= 90:
                errors_last_10 += err
            p.update(pattern[t])
        # After training on the pattern, errors should decrease
        assert errors_last_10 <= errors_first_10

    def test_predict_uses_strict_greater_than_threshold(self):
        p = SpikePredictor(n_channels=3, history_len=2, threshold=0.5, seed=7)
        p.W[:] = 0.0
        p.bias[:] = 0.0  # sigmoid(0) == 0.5 exactly
        pred = p.predict()
        np.testing.assert_array_equal(pred, np.zeros(3, dtype=np.int8))

    def test_reset_clears_history_and_time_counter(self):
        p = SpikePredictor(n_channels=2, history_len=3, seed=11)
        p.update(np.array([1, 0], dtype=np.int8))
        p.update(np.array([0, 1], dtype=np.int8))
        assert p._t == 2
        assert np.any(p._history != 0.0)
        p.reset()
        assert p._t == 0
        assert np.all(p._history == 0.0)
