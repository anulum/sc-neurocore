# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExplanationDiff from former test_explainability.py

"""Focused suite: TestExplanationDiff from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403


class TestExplanationDiff:
    def test_identical_nodes_no_diffs(self):
        tree = SpikeDecisionTree()
        bs = np.ones(8, dtype=np.uint8)
        a = tree.add_decision("n0", bs, threshold=4)
        tree2 = SpikeDecisionTree()
        b = tree2.add_decision("n0", bs, threshold=4)
        diffs = ExplanationDiff.diff(a, b)
        assert diffs == []

    def test_different_thresholds(self):
        tree = SpikeDecisionTree()
        bs = np.ones(8, dtype=np.uint8)
        a = tree.add_decision("n0", bs, threshold=4)
        b = tree.add_decision("n1", bs, threshold=6)
        diffs = ExplanationDiff.diff(a, b)
        fields_changed = [d.field for d in diffs]
        assert "neuron_id" in fields_changed
        assert "threshold" in fields_changed

    def test_different_decisions(self):
        tree = SpikeDecisionTree()
        a = tree.add_decision("n0", np.ones(8, dtype=np.uint8), threshold=4)
        b = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), threshold=4)
        diffs = ExplanationDiff.diff(a, b)
        fields_changed = [d.field for d in diffs]
        assert "decision" in fields_changed
