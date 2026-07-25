# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Local LLM HTTP transport tests

"""Validate loopback transport policy and HTTP failure translation."""

from __future__ import annotations

import io
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from sc_neurocore.bridges.local_llm import (
    LocalLLMBridge,
    LocalLLMConfig,
    LocalLLMError,
    LocalLLMProvider,
)
from tests.test_bridges.local_llm_support import _FakeHTTPResponse, _patch_urlopen


def test_post_json_timeout_raises_local_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA, timeout_sec=0.01))

    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise TimeoutError("slow local model")

    _patch_urlopen(monkeypatch, fake_urlopen)
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


def test_post_json_returns_decoded_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful POST decodes and returns the JSON object body."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))

    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(b'{"message": {"content": "ok"}}')

    _patch_urlopen(monkeypatch, fake_urlopen)

    assert bridge._post_json({"model": "m", "messages": []}) == {"message": {"content": "ok"}}


def test_post_json_wraps_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An HTTPError is surfaced as a LocalLLMError including the response body."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))

    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        headers = Message()
        raise HTTPError(
            "http://127.0.0.1",
            500,
            "boom",
            headers,
            io.BytesIO(b"server detail"),
        )

    _patch_urlopen(monkeypatch, fake_urlopen)

    with pytest.raises(LocalLLMError, match="HTTP error 500: server detail"):
        bridge._post_json({"model": "m", "messages": []})


def test_post_json_wraps_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URLError is surfaced as an endpoint-unavailable LocalLLMError."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))

    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise URLError("connection refused")

    _patch_urlopen(monkeypatch, fake_urlopen)

    with pytest.raises(LocalLLMError, match="endpoint unavailable: connection refused"):
        bridge._post_json({"model": "m", "messages": []})


def test_post_json_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JSON body is rejected as invalid JSON."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))
    _patch_urlopen(monkeypatch, lambda *_args, **_kwargs: _FakeHTTPResponse(b"not json"))

    with pytest.raises(LocalLLMError, match="invalid JSON"):
        bridge._post_json({"model": "m", "messages": []})


def test_post_json_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON array body is rejected because a JSON object is required."""
    bridge = LocalLLMBridge(LocalLLMConfig(provider=LocalLLMProvider.OLLAMA))
    _patch_urlopen(monkeypatch, lambda *_args, **_kwargs: _FakeHTTPResponse(b"[1, 2, 3]"))

    with pytest.raises(LocalLLMError, match="non-object JSON"):
        bridge._post_json({"model": "m", "messages": []})
