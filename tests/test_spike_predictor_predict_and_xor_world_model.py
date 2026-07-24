# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPredictAndXorWorldModel from former test_spike_predictor.py

"""Focused suite: TestPredictAndXorWorldModel from former test_spike_predictor.py."""

from __future__ import annotations

from tests.spike_predictor_support import *  # noqa: F403


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
