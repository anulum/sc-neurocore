# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCCInfluence from former test_explainability.py

"""Focused suite: TestSCCInfluence from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403


class TestSCCInfluence:
    def test_influence_computed(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50, scc=0.5)
        assert node.scc_influence > 0

    def test_zero_scc_zero_influence(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50, scc=0.0)
        assert node.scc_influence == 0.0

    def test_influence_in_dict(self):
        tree = SpikeDecisionTree()
        bs = np.ones(8, dtype=np.uint8)
        tree.add_decision("n0", bs, threshold=4, scc=0.3)
        d = tree.to_dict()
        assert "scc_influence" in d
        assert "margin" in d
        assert "confidence" in d
