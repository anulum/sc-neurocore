# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for world model spike predictor

import numpy as np

from sc_neurocore.world_model.spike_predictor import (
    SpikePredictor,
    predict_and_xor_world_model,
    xor_and_recover_world_model,
)


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


class TestCodecRoundtrip:
    def test_lossless_roundtrip(self):
        n_ch = 4
        T = 20
        rng = np.random.RandomState(42)
        spikes = (rng.random((T, n_ch)) < 0.3).astype(np.int8)

        errors, correct = predict_and_xor_world_model(spikes, n_channels=n_ch, seed=0)
        recovered = xor_and_recover_world_model(errors, n_channels=n_ch, seed=0)
        np.testing.assert_array_equal(spikes, recovered)

    def test_correct_count_sane(self):
        n_ch = 4
        T = 30
        rng = np.random.RandomState(0)
        spikes = (rng.random((T, n_ch)) < 0.3).astype(np.int8)
        _, correct = predict_and_xor_world_model(spikes, n_channels=n_ch)
        assert 0 <= correct <= T * n_ch

    def test_errors_are_binary(self):
        n_ch = 2
        T = 20
        spikes = np.zeros((T, n_ch), dtype=np.int8)
        spikes[::2, 0] = 1
        errors, _ = predict_and_xor_world_model(spikes, n_channels=n_ch)
        assert set(np.unique(errors)).issubset({0, 1})
