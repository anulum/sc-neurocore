# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the local LLM bridge

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.bridges.local_llm import (
    LocalLLMBridge,
    LocalLLMConfig,
    LocalLLMError,
    LocalLLMProvider,
    SpikePromptAdapter,
)


def test_auto_provider_prefers_ollama_for_default_port() -> None:
    cfg = LocalLLMConfig()
    assert cfg.resolved_provider() is LocalLLMProvider.OLLAMA


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


def test_post_json_timeout_raises_local_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA, timeout_sec=0.01))

    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise TimeoutError("slow local model")

    monkeypatch.setattr("sc_neurocore.bridges.local_llm.urlopen", fake_urlopen)
    with pytest.raises(LocalLLMError, match="timed out"):
        bridge._post_json({"model": "llama3:latest", "messages": []})


def test_post_json_rejects_non_http_scheme() -> None:
    bridge = LocalLLMBridge(
        LocalLLMConfig(
            base_url="file:///tmp/local-model.sock", provider=LocalLLMProvider.CHAT_COMPLETIONS
        )
    )
    with pytest.raises(LocalLLMError, match="http or https"):
        bridge._post_json({"model": "local-chat", "messages": []})


def test_post_json_rejects_non_loopback_host() -> None:
    bridge = LocalLLMBridge(
        LocalLLMConfig(base_url="http://10.0.0.5:11434", provider=LocalLLMProvider.OLLAMA)
    )
    with pytest.raises(LocalLLMError, match="loopback-local"):
        bridge._post_json({"model": "llama3:latest", "messages": []})


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
    ):
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


def test_auto_provider_falls_back_to_chat_completions_for_generic_host() -> None:
    """A non-Ollama base URL resolves AUTO to the chat-completions provider."""
    config = LocalLLMConfig(base_url="http://localhost:8000", provider=LocalLLMProvider.AUTO)

    assert config.resolved_provider() is LocalLLMProvider.CHAT_COMPLETIONS


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


def test_endpoint_rejects_unresolved_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endpoint built from an unresolved AUTO provider raises a clear error."""
    monkeypatch.setattr(LocalLLMConfig, "resolved_provider", lambda self: LocalLLMProvider.AUTO)
    bridge = LocalLLMBridge(LocalLLMConfig())

    with pytest.raises(LocalLLMError, match="Unsupported provider"):
        bridge._endpoint()


class _FakeHTTPResponse:
    """Context-manager stand-in for an urlopen response with a JSON body."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    monkeypatch.setattr("sc_neurocore.bridges.local_llm.urlopen", handler)


def test_post_json_returns_decoded_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful POST decodes and returns the JSON object body."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))

    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(b'{"message": {"content": "ok"}}')

    _patch_urlopen(monkeypatch, fake_urlopen)

    assert bridge._post_json({"model": "m", "messages": []}) == {"message": {"content": "ok"}}


def test_post_json_wraps_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An HTTPError is surfaced as a LocalLLMError including the response body."""
    import io
    from urllib.error import HTTPError

    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))

    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise HTTPError("http://127.0.0.1", 500, "boom", {}, io.BytesIO(b"server detail"))  # type: ignore[arg-type]

    _patch_urlopen(monkeypatch, fake_urlopen)

    with pytest.raises(LocalLLMError, match="HTTP error 500: server detail"):
        bridge._post_json({"model": "m", "messages": []})


def test_post_json_wraps_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URLError is surfaced as an endpoint-unavailable LocalLLMError."""
    from urllib.error import URLError

    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))

    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise URLError("connection refused")

    _patch_urlopen(monkeypatch, fake_urlopen)

    with pytest.raises(LocalLLMError, match="endpoint unavailable: connection refused"):
        bridge._post_json({"model": "m", "messages": []})


def test_post_json_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JSON body is rejected as invalid JSON."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))
    _patch_urlopen(monkeypatch, lambda *_a, **_k: _FakeHTTPResponse(b"not json"))

    with pytest.raises(LocalLLMError, match="invalid JSON"):
        bridge._post_json({"model": "m", "messages": []})


def test_post_json_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON array body is rejected because a JSON object is required."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))
    _patch_urlopen(monkeypatch, lambda *_a, **_k: _FakeHTTPResponse(b"[1, 2, 3]"))

    with pytest.raises(LocalLLMError, match="non-object JSON"):
        bridge._post_json({"model": "m", "messages": []})


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
