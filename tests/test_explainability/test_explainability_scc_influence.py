# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCCInfluence from former test_explainability.py

"""Focused suite: TestSCCInfluence from former test_explainability.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.explainability.explainability import (
    SpikeDecisionTree,
)


class TestSCCInfluence:
    """The SCC influence of a decision."""

    def test_influence_computed(self) -> None:
        """Correlated input gives a positive influence."""
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50, scc=0.5)
        assert node.scc_influence > 0

    def test_zero_scc_zero_influence(self) -> None:
        """A zero SCC gives a zero influence."""
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50, scc=0.0)
        assert node.scc_influence == 0.0

    def test_influence_in_dict(self) -> None:
        """The node exports its influence, margin and confidence."""
        tree = SpikeDecisionTree()
        bs = np.ones(8, dtype=np.uint8)
        tree.add_decision("n0", bs, threshold=4, scc=0.3)
        d = tree.to_dict()
        assert "scc_influence" in d
        assert "margin" in d
        assert "confidence" in d
