# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Local LLM spike-prompt adapter tests

"""Validate spike summaries and their delivery to the local-LLM bridge."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.bridges.local_llm import (
    LocalLLMBridge,
    LocalLLMConfig,
    LocalLLMProvider,
    LocalLLMResponse,
    SpikePromptAdapter,
)


def test_raster_summary_reports_density_and_top_neurons() -> None:
    raster = np.array(
        [
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    text = SpikePromptAdapter.raster_summary(raster, dt_ms=2.0, neuron_labels=["a", "b", "c"])
    assert "timesteps=4" in text
    assert "neurons=3" in text
    assert "density=" in text
    assert "- c:" in text


def test_raster_summary_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="shape"):
        SpikePromptAdapter.raster_summary(np.array([1, 0, 1], dtype=bool))


def test_analyse_spike_raster_forwards_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.CHAT_COMPLETIONS))
    captured_prompt: dict[str, str] = {}
    original_chat = bridge.chat

    def fake_chat(
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LocalLLMResponse:
        del system_prompt, model, temperature, max_tokens
        captured_prompt["text"] = user_prompt
        return original_chat("test-response")

    def fake_post(_payload: dict[str, object]) -> dict[str, object]:
        return {
            "model": "local-chat",
            "choices": [
                {"message": {"role": "assistant", "content": "analysis"}, "finish_reason": "stop"}
            ],
            "usage": {},
        }

    monkeypatch.setattr(bridge, "_post_json", fake_post)
    monkeypatch.setattr(bridge, "chat", fake_chat)
    raster = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    result = LocalLLMBridge.analyse_spike_raster(
        bridge,
        raster,
        question="Describe this activity.",
        neuron_labels=["x", "y"],
    )
    assert result.text == "analysis"
    assert "Describe this activity." in captured_prompt["text"]
    assert "Spike summary:" in captured_prompt["text"]


def test_summarise_rates_handles_empty_input() -> None:
    """An empty rate vector yields an explicit no-neurons message."""
    assert SpikePromptAdapter.summarise_rates(np.zeros(0)) == "No neurons were provided."


def test_raster_summary_handles_empty_raster() -> None:
    """A raster with a zero-length axis is reported as empty."""
    assert SpikePromptAdapter.raster_summary(np.zeros((0, 3))) == "Empty spike raster."


def test_raster_summary_rejects_non_positive_dt() -> None:
    """A non-positive timestep is rejected because it yields a zero duration."""
    with pytest.raises(ValueError, match="dt_ms must be positive"):
        SpikePromptAdapter.raster_summary(np.ones((2, 3)), dt_ms=0.0)
