# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEngineIntegration from former test_explainability.py

"""Focused suite: TestEngineIntegration from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403


class TestEngineIntegration:
    def test_temporal_tracking(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20, timestep=0)
        engine.explain_spike("n1", 32768, 64, 20, timestep=1)
        assert engine.temporal.num_timesteps == 2

    def test_multi_layer_tracking(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20, layer_id="L1")
        engine.explain_spike("n1", 32768, 64, 20, layer_id="L2")
        assert len(engine.multi_layer.layer_ids) == 2

    def test_symbolic_path_tracking(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20)
        assert engine.symbolic.length == 1

    def test_explain_spike_with_local_llm(self, monkeypatch):
        engine = ExplainabilityEngine(seed=0xACE1)
        bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.CHAT_COMPLETIONS))

        def fake_chat(user_prompt: str, **_kwargs):
            assert "Base explanation:" in user_prompt

            class _Resp:
                text = "Enhanced local explanation"

            return _Resp()

        monkeypatch.setattr(bridge, "chat", fake_chat)
        node, text = engine.explain_spike_with_local_llm(
            "n0",
            32768,
            64,
            20,
            bridge=bridge,
        )
        assert node.neuron_id == "n0"
        assert text == "Enhanced local explanation"
        lst = engine.symbolic.to_list()
        assert "popcount" in lst[0]["reason"]

    def test_to_dict_has_all_fields(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike(
            "n0", 32768, 64, 20, scc=0.2, layer_id="L1", timestep=3, contributing_neurons=["src0"]
        )
        d = engine.tree.to_dict()
        assert "scc_influence" in d
        assert "margin" in d
        assert "confidence" in d
        assert "timestep" in d
        assert "layer_id" in d
        assert "contributing_neurons" in d
