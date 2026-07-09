# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.world_model.spike_predictor

from __future__ import annotations

import numpy as np

from sc_neurocore.world_model.spike_predictor import (
    SpikePredictor,
    predict_and_xor_world_model,
    xor_and_recover_world_model,
)


class TestSpikePredictor:
    def test_predict_shape(self):
        p = SpikePredictor(n_channels=8, history_len=4)
        pred = p.predict()
        assert pred.shape == (8,)
        assert set(np.unique(pred).tolist()) <= {0, 1}

    def test_predict_probs_bounded(self):
        p = SpikePredictor(n_channels=4, history_len=4)
        probs = p.predict_probs()
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_update_changes_predictions(self):
        p = SpikePredictor(n_channels=4, history_len=4, lr=0.1)
        pred_before = p.predict_probs().copy()
        # Feed all-ones for several steps
        for _ in range(10):
            p.update(np.ones(4, dtype=np.int8))
        pred_after = p.predict_probs()
        # Predictions should have moved toward higher values
        assert np.mean(pred_after) > np.mean(pred_before)

    def test_reset_restores_state(self):
        p = SpikePredictor(n_channels=4, history_len=4, seed=42)
        w_init = p.W.copy()
        for _ in range(10):
            p.update(np.ones(4, dtype=np.int8))
        p.reset()
        np.testing.assert_array_equal(p.W, w_init)

    def test_deterministic(self):
        p1 = SpikePredictor(n_channels=4, history_len=4, seed=99)
        p2 = SpikePredictor(n_channels=4, history_len=4, seed=99)
        data = np.array([1, 0, 1, 0], dtype=np.int8)
        for _ in range(5):
            p1.update(data)
            p2.update(data)
        np.testing.assert_array_equal(p1.predict(), p2.predict())


class TestPredictAndXorWorldModel:
    def test_roundtrip(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 8)) < 0.05).astype(np.int8)
        errors, _ = predict_and_xor_world_model(spikes, 8, seed=42)
        recovered = xor_and_recover_world_model(errors, 8, seed=42)
        np.testing.assert_array_equal(recovered, spikes)

    def test_silent_roundtrip(self):
        spikes = np.zeros((100, 4), dtype=np.int8)
        errors, correct = predict_and_xor_world_model(spikes, 4)
        assert correct == 400  # all correct
        recovered = xor_and_recover_world_model(errors, 4)
        np.testing.assert_array_equal(recovered, spikes)

    def test_accuracy_improves_on_pattern(self):
        """Repeated pattern should be learned → accuracy increases."""
        rng = np.random.RandomState(42)
        T, N = 500, 4
        pattern = np.array([1, 0, 1, 0], dtype=np.int8)
        spikes = np.tile(pattern, (T, 1))
        _, correct = predict_and_xor_world_model(spikes, N, lr=0.05, seed=42)
        accuracy = correct / (T * N)
        # Should learn the constant pattern → high accuracy
        assert accuracy > 0.8
