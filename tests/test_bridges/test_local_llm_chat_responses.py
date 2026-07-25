# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Local LLM chat response parsing tests

"""Validate Ollama and chat-completions response parsing."""

from __future__ import annotations

import pytest

from sc_neurocore.bridges.local_llm import (
    LocalLLMBridge,
    LocalLLMConfig,
    LocalLLMError,
    LocalLLMProvider,
)


def test_chat_ollama_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA, model="local-ollama"))

    def fake_post(payload: dict[str, object]) -> dict[str, object]:
        assert payload["model"] == "local-ollama"
        assert payload["stream"] is False
        return {
            "model": "local-ollama",
            "message": {"role": "assistant", "content": "local answer"},
            "prompt_eval_count": 17,
            "eval_count": 9,
            "done_reason": "stop",
        }

    monkeypatch.setattr(bridge, "_post_json", fake_post)
    result = bridge.chat("hello")
    assert result.text == "local answer"
    assert result.model == "local-ollama"
    assert result.prompt_tokens == 17
    assert result.completion_tokens == 9


def test_chat_completions_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = LocalLLMBridge(
        LocalLLMConfig(
            base_url="http://127.0.0.1:8000",
            provider=LocalLLMProvider.CHAT_COMPLETIONS,
            model="local-chat",
        )
    )

    def fake_post(payload: dict[str, object]) -> dict[str, object]:
        assert payload["model"] == "local-chat"
        return {
            "model": "local-chat",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "compat answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 21, "completion_tokens": 11},
        }

    monkeypatch.setattr(bridge, "_post_json", fake_post)
    result = bridge.chat("hello")
    assert result.text == "compat answer"
    assert result.model == "local-chat"
    assert result.prompt_tokens == 21
    assert result.completion_tokens == 11


def test_chat_rejects_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))

    def fake_post(_payload: dict[str, object]) -> dict[str, object]:
        return {"model": "broken"}

    monkeypatch.setattr(bridge, "_post_json", fake_post)
    with pytest.raises(LocalLLMError, match="message.content"):
        bridge.chat("hello")


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"choices": []}, "missing choices"),
        ({"choices": [123]}, "choice is malformed"),
        ({"choices": [{"finish_reason": "stop"}]}, "missing choices\\[0\\].message.content"),
    ],
)
def test_chat_completions_rejects_malformed_choices(
    monkeypatch: pytest.MonkeyPatch, raw: dict[str, object], match: str
) -> None:
    """The chat-completions parser rejects missing, malformed and contentless choices."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.CHAT_COMPLETIONS))
    monkeypatch.setattr(bridge, "_post_json", lambda _payload: raw)

    with pytest.raises(LocalLLMError, match=match):
        bridge.chat("hello")
