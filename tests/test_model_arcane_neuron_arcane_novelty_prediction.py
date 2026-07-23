# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcaneNoveltyPrediction from former test_model_arcane_neuron.py

"""Focused suite: TestArcaneNoveltyPrediction from former test_model_arcane_neuron.py."""

from __future__ import annotations

from tests.model_arcane_neuron_support import *  # noqa: F403

class TestArcaneNoveltyPrediction:
    """Self-referential: predict own state → surprise → novelty."""

    def test_surprise_computed(self):
        n = ArcaneNeuron()
        n.step(2.0)
        assert n._surprise >= 0

    def test_novelty_sigmoid(self):
        """novelty = sigmoid(κ·(surprise - baseline)). Bounded [0, 1]."""
        n = ArcaneNeuron()
        for _ in range(100):
            n.step(2.0)
        assert 0 <= n._novelty <= 1

    def test_predictor_weights_normalised(self):
        """w_pred is normalised after each update."""
        n = ArcaneNeuron()
        for _ in range(1000):
            n.step(2.0)
        norm = np.linalg.norm(n.w_pred)
        assert abs(norm - 1.0) < 0.01 or norm == 0

    def test_meta_lr_increases_with_novelty(self):
        """meta_lr = lr_base * (1 + η * novelty). Higher novelty → faster learning."""
        n = ArcaneNeuron()
        n._novelty = 0.0
        lr_low = n.meta_learning_rate
        n._novelty = 1.0
        lr_high = n.meta_learning_rate
        assert lr_high > lr_low

    def test_confidence_decreases_with_novelty(self):
        """confidence = 1 - mean(novelty_history). High novelty → low confidence."""
        n = ArcaneNeuron()
        n._novelty_history = [0.9] * 20
        n.step(0.0)  # updates confidence
        assert n._confidence < 0.2
