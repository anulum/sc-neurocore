# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDecisionMargin from former test_explainability.py

"""Focused suite: TestDecisionMargin from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403

class TestDecisionMargin:
    def test_spike_margin_positive(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=80)
        m = node.margin
        assert m.margin == 20
        assert m.confidence > 0

    def test_no_spike_margin_negative(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:30] = 1
        node = tree.add_decision("n0", bs, threshold=50)
        m = node.margin
        assert m.margin == -20
        assert m.confidence > 0

    def test_exact_threshold(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:50] = 1
        node = tree.add_decision("n0", bs, threshold=50)
        assert node.margin.margin == 0
        assert node.decision == SpikeDecision.SPIKE
