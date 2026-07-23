# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNaturalLanguageExplainer from former test_explainability.py

"""Focused suite: TestNaturalLanguageExplainer from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403

class TestNaturalLanguageExplainer:
    def test_explain_spike(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision(
            "n0", bs, threshold=50, scc=0.3, contributing_neurons=["in_a", "in_b"]
        )
        text = NaturalLanguageExplainer.explain_node(node)
        assert "fired" in text
        assert "n0" in text
        assert "SCC" in text
        assert "in_a" in text

    def test_explain_no_spike(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50)
        text = NaturalLanguageExplainer.explain_node(node)
        assert "NOT" in text

    def test_explain_attribution(self):
        attr = CausalAttribution("out", {"a": 80.0, "b": 20.0}, 100.0)
        text = NaturalLanguageExplainer.explain_attribution(attr)
        assert "out" in text
        assert "a" in text

    def test_explain_sensitivity_robust(self):
        results = [
            SensitivityResult("n0", 50, 49, SpikeDecision.SPIKE, SpikeDecision.SPIKE, False),
            SensitivityResult("n0", 50, 51, SpikeDecision.SPIKE, SpikeDecision.SPIKE, False),
        ]
        text = NaturalLanguageExplainer.explain_sensitivity(results)
        assert "robust" in text

    def test_explain_sensitivity_flip(self):
        results = [
            SensitivityResult("n0", 50, 51, SpikeDecision.SPIKE, SpikeDecision.NO_SPIKE, True),
        ]
        text = NaturalLanguageExplainer.explain_sensitivity(results)
        assert "flip" in text

    def test_explain_node_with_local_llm(self, monkeypatch):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50)
        bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.CHAT_COMPLETIONS))

        def fake_chat(user_prompt: str, **_kwargs):
            assert "Base explanation:" in user_prompt

            class _Resp:
                text = "Local rewrite"

            return _Resp()

        monkeypatch.setattr(bridge, "chat", fake_chat)
        text = NaturalLanguageExplainer.explain_node_with_local_llm(node, bridge=bridge)
        assert text == "Local rewrite"
