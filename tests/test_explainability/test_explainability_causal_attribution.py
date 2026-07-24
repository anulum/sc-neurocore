# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCausalAttribution from former test_explainability.py

"""Focused suite: TestCausalAttribution from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403


class TestCausalAttribution:
    def test_basic_attribution(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        target = tree.add_decision("out", bs, threshold=50)
        inputs = {
            "in0": np.ones(100, dtype=np.uint8),
            "in1": np.zeros(100, dtype=np.uint8),
        }
        attr = CausalAttributor.attribute(target, inputs)
        assert attr.target_neuron == "out"
        assert attr.attributions["in0"] == 100.0
        assert attr.attributions["in1"] == 0.0
        assert attr.total_contribution == 100.0

    def test_weighted_attribution(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        target = tree.add_decision("out", bs, threshold=50)
        inputs = {"in0": np.ones(50, dtype=np.uint8)}
        weights = {"in0": 2.0}
        attr = CausalAttributor.attribute(target, inputs, weights)
        assert attr.attributions["in0"] == 100.0

    def test_top_contributors_sorted(self):
        tree = SpikeDecisionTree()
        target = tree.add_decision("out", np.ones(8, dtype=np.uint8), 4)
        inputs = {
            "a": np.ones(10, dtype=np.uint8),
            "b": np.zeros(10, dtype=np.uint8),
            "c": np.ones(5, dtype=np.uint8),
        }
        attr = CausalAttributor.attribute(target, inputs)
        top = attr.top_contributors
        assert top[0][0] == "a"
        assert top[0][1] > top[1][1]

    def test_engine_attribute(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        node = engine.explain_spike("n0", 32768, 64, 20)
        inputs = {"src0": np.ones(64, dtype=np.uint8)}
        attr = engine.attribute(node, inputs)
        assert attr.total_contribution > 0
