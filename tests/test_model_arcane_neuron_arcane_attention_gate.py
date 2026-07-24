# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcaneAttentionGate from former test_model_arcane_neuron.py

"""Focused suite: TestArcaneAttentionGate from former test_model_arcane_neuron.py."""

from __future__ import annotations

from tests.model_arcane_neuron_support import *  # noqa: F403


class TestArcaneAttentionGate:
    def test_gate_sigmoid(self):
        """gate = sigmoid(w_g · [I, v_fast, v_work, confidence]). Bounded (0, 1)."""
        n = ArcaneNeuron()
        # Gate output is internal — verify indirectly: higher I → more v_fast
        n_low = ArcaneNeuron(theta=100.0)
        n_high = ArcaneNeuron(theta=100.0)
        n_low.step(0.5)
        n_high.step(5.0)
        assert n_high.v_fast > n_low.v_fast

    def test_gate_modulates_effective_input(self):
        """Gate filters input before fast compartment."""
        # With zero gate weights → no input passes
        n = ArcaneNeuron(theta=100.0)
        n.w_gate = np.array([0.0, 0.0, 0.0, 0.0])
        n.step(10.0)
        # gate = sigmoid(0) = 0.5, i_eff = 0.5 * 10 = 5.0
        # v_fast should have changed
        assert n.v_fast > 0  # sigmoid(0) = 0.5, so some input gets through
