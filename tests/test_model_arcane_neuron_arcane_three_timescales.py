# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcaneThreeTimescales from former test_model_arcane_neuron.py

"""Focused suite: TestArcaneThreeTimescales from former test_model_arcane_neuron.py."""

from __future__ import annotations

from tests.model_arcane_neuron_support import *  # noqa: F403

class TestArcaneThreeTimescales:
    def test_fast_fastest(self):
        """v_fast (τ=5) changes fastest."""
        n = ArcaneNeuron(theta=100.0)  # prevent spikes
        n.step(2.0)
        assert abs(n.v_fast) > abs(n.v_work)
        assert abs(n.v_fast) > abs(n.v_deep)

    def test_working_memory_on_spike(self):
        """v_work updates only when spike occurs (gated by spike)."""
        n = ArcaneNeuron()
        for _ in range(5000):
            if n.step(2.0) == 1:
                break
        assert n.v_work > 0, "v_work should update after spike"
