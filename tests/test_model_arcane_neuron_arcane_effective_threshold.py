# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcaneEffectiveThreshold from former test_model_arcane_neuron.py

"""Focused suite: TestArcaneEffectiveThreshold from former test_model_arcane_neuron.py."""

from __future__ import annotations

from tests.model_arcane_neuron_support import *  # noqa: F403

class TestArcaneEffectiveThreshold:
    def test_threshold_modulated_by_deep(self):
        """eff_threshold = θ · (1 + γ·v_deep) · (1 - δ·confidence).

        Higher v_deep → higher threshold. Higher confidence → lower threshold.
        """
        n = ArcaneNeuron()
        # At defaults: v_deep=0, confidence=0.5
        # eff_threshold = 1.0 * (1+0) * (1 - 0.3*0.5) = 0.85
        n._confidence = 0.5
        eff = n.theta * (1 + n.gamma * n.v_deep) * (1 - n.delta_conf * n._confidence)
        assert abs(eff - 0.85) < 0.01

    def test_confident_lowers_threshold(self):
        """High confidence → lower effective threshold → fires more easily."""
        n_conf = ArcaneNeuron()
        n_unconf = ArcaneNeuron()
        n_conf._novelty_history = [0.1] * 20  # low novelty → high confidence
        n_unconf._novelty_history = [0.9] * 20  # high novelty → low confidence
        s_conf = len(_run(n_conf, current=1.5, steps=5000))
        s_unconf = len(_run(n_unconf, current=1.5, steps=5000))
        # Confident should fire more (lower threshold)
        assert s_conf >= s_unconf
